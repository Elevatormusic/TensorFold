"""DeepSeek-V4.1 host-memory guards: the blocked Engram sum, the cache trim and the admission gate."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.memory_gate import NoRoom  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import model, multi  # noqa: E402


class Peer:
    """Rank 0 of two: the other rank's copy is 3x this one's."""

    def __init__(self):
        self.calls = 0

    def all_gather(self, send, recv):
        self.calls += 1
        recv.view(2, -1)[0].copy_(send)
        recv.view(2, -1)[1].copy_(send * 3)


@pytest.mark.parametrize("n,block", [(1300, 512), (1024, 512), (7, 512), (2048, 0), (2049, 700)])
def test_sum_blocks_equals_the_whole_sum(n, block):
    x = torch.randn(n, 37)
    whole = model.Comm(Peer(), 2).sum(x).to(torch.bfloat16)
    peer = Peer()
    got = model.Comm(peer, 2).sum_blocks(x, block, torch.bfloat16)
    assert got.dtype == torch.bfloat16 and torch.equal(got, whole)
    assert peer.calls == (1 if block <= 0 or n <= block else -(-n // block))


def test_sum_blocks_on_one_rank_is_the_input_cast():
    x = torch.randn(10, 4)
    assert torch.equal(model.Comm(Peer(), 1).sum_blocks(x, 4, torch.bfloat16), x.to(torch.bfloat16))


def _decoder(live, room):
    return SimpleNamespace(live=lambda: live, _host_room=lambda: room, gate_log=0.0)


GIB = 2**30


def test_gate_waits_only_with_live_streams_and_a_short_host(monkeypatch, capsys):
    monkeypatch.setenv("TF_DS_MEM_FLOOR_GIB", "1.5")
    monkeypatch.setattr(multi, "ADMIT_GIB", 1.0)
    multi.MultiDecoder._gate(_decoder(2, 3 * GIB), None)                     # room: goes in
    multi.MultiDecoder._gate(_decoder(0, 1 * GIB), None)                     # nothing live: waiting would not help
    multi.MultiDecoder._gate(_decoder(2, -1), None)                          # unknown: goes in
    short = _decoder(2, 2 * GIB)
    with pytest.raises(NoRoom):
        multi.MultiDecoder._gate(short, None)
    assert "admission waits" in capsys.readouterr().out
    with pytest.raises(NoRoom):                                              # (no second line within 10 s)
        multi.MultiDecoder._gate(short, None)
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(multi, "ADMIT_GIB", 0.0)
    multi.MultiDecoder._gate(short, None)                                    # off


def test_trim_is_off_below_its_threshold(monkeypatch):
    freed = []
    monkeypatch.setattr(multi.torch.cuda, "empty_cache", lambda: freed.append(1))
    monkeypatch.setattr(multi.torch.cuda, "memory_reserved", lambda *a: 1000 * 2**20)
    monkeypatch.setattr(multi.torch.cuda, "memory_allocated", lambda *a: 800 * 2**20)
    monkeypatch.setattr(multi, "TRIM", 256 * 2**20)
    multi.MultiDecoder._trim(None)
    assert freed == []
    monkeypatch.setattr(multi.torch.cuda, "memory_allocated", lambda *a: 700 * 2**20)
    multi.MultiDecoder._trim(None)
    assert freed == [1]
    monkeypatch.setattr(multi, "TRIM", -1)
    multi.MultiDecoder._trim(None)
    assert freed == [1]
