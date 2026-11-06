"""Chunk order on simulated ranks. This does not test two-node hardware or prompt latency."""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")

from test_deepseek_v41_spill import CHUNK, mods, _decoder, _prompt


@pytest.mark.parametrize("stop", ["count", "callback"])
def test_both_ranks_rotate_unfinished_prompts_after_each_chunk(mods, monkeypatch, stop):
    multi, spill, root = mods
    monkeypatch.setattr(multi, "QUICK_ROWS", 0)
    decoders = [_decoder(multi, spill, root / str(rank))[0] for rank in range(2)]
    for md in decoders:
        md.slack0, md.gate_log = None, 0.0
        for seed in (1, 2):
            count = 10 if seed == 1 and stop == "callback" else 1
            s = multi.Stream(_prompt(CHUNK * 3 + 1, seed), count, None, draft=False, stop_eos=False)
            s.emit = (lambda tokens: True) if seed == 1 and stop == "callback" else (lambda tokens: None)
            md.admit(s)
    leader, follower = decoders
    digests = [[], []]
    for rank, md in enumerate(decoders):
        md.e.rank, md.e.world = rank, 2
        md._agree = lambda label, plan, rank=rank: digests[rank].append((label, plan))
    sent = []
    leader._send = lambda op: sent.append(op)
    for _ in range(6):
        leader.round()
        op = sent[-1]
        follower._fill(next(s for s in follower.filling if s.sid == op[1]))
        assert [(s.sid, s.filled) for s in leader.filling] == [(s.sid, s.filled) for s in follower.filling]
        assert digests[0] == digests[1]
    assert sent == [["fill", 0], ["fill", 1]] * 3
    # Completion removes the stream; a completed request cannot take another chunk.
    assert leader.round()[0].sid == 0
    assert [s.sid for s in leader.filling] == [1]
    assert leader.round()[0].sid == 1
    assert not leader.filling


def test_decode_quota_and_one_chunk_quick_priority_are_kept(mods, monkeypatch):
    multi, spill, root = mods
    md, _ = _decoder(multi, spill, root)
    long = NS(sid=0, prompt=[0] * (3 * CHUNK), filled=0)
    quick = NS(sid=1, prompt=[0] * CHUNK, filled=0)
    md.filling = [long, quick]
    md.streams = {5: NS(done=False)}
    sent, filled, decoded = [], [], []
    md._send = lambda op: sent.append(op)
    md._fill = lambda s: filled.append(s.sid) or []
    md._depth = lambda n: 0
    md._round = lambda live, k: decoded.append(len(live)) or []
    md.chunk_s, md.since_fill = 1.0, 0.0
    monkeypatch.setattr(multi, "QUICK_ROWS", CHUNK * 8)
    assert not md._quick(long)                              # quick priority is at most one chunk
    md.round()
    assert filled == [1] and not decoded                    # a short prompt can finish before the decode quota
    md.filling = [long]
    md.round()
    assert decoded == [1] and filled == [1]                 # a long prompt does not bypass the decode quota
    md.since_fill = 100.0
    md.round()
    assert filled == [1, 0]
