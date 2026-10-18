"""Structured output on the DeepSeek-V4.1 engine (families/deepseek_v41/cuda/constrain.py): a reply's grammar over a
DSpark draft chain cuts the drafts it cannot keep, masks the verify rows and follows the kept tokens, and a follower
rank's copy (rank 0's ``pack``, compiled there) masks bit for bit the same, and a reply ends at its grammar's stop
token also when the request ignores the end token (real xgrammar, a tiny vocabulary)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
xgr = pytest.importorskip("xgrammar")

from tensorfold.cuda.streams import Stream
from tensorfold.engine.grammar import Grammars, Spec, pack
from tensorfold.families.deepseek_v41.cuda import constrain, engine, multi

VOCAB = ["<eos>", "a", "b", "x", "</think>", "y", "e", "s", "n", "o", "{", "}", '"', ":", "1", " "]
ID = {t: i for i, t in enumerate(VOCAB)}
EOS, END = ID["<eos>"], ID["</think>"]
WIDTH = 20                                         # logits wider than the vocabulary: the extra columns never allowed


def grammars():
    return Grammars(xgr.TokenizerInfo(VOCAB, vocab_size=WIDTH, stop_token_ids=[EOS]))


def constraint(spec, think_end=None):
    g = grammars()
    return g.constraint(g.compile(spec), think_end=think_end, spec=spec)


def ids(*tokens):
    return [ID[t] for t in tokens]


def allowed(logits):
    return [[VOCAB[i] for i in range(len(VOCAB)) if row[i] > float("-inf")] for row in logits.tolist()]


class Rank:                                        # a follower rank: ``constrain.follow`` compiles with its compiler
    def __init__(self):
        self.grammars, self.model_dir, self.eos = grammars(), "unused", (EOS,)


def test_without_a_grammar_nothing_changes():
    assert constrain.chain(None, 1, [2, 3]) == ([2, 3], None)
    logits = torch.zeros(3, WIDTH)
    constrain.mask(None, logits, None)
    constrain.first(None, logits)
    constrain.advance(None, [1])
    assert bool((logits == 0).all())
    assert constrain.follow(Rank(), []) is None


def test_the_first_token_then_a_chain_cut_masked_and_followed():
    c = constraint(Spec("regex", "ab+"))
    first = torch.zeros(1, WIDTH)
    constrain.first(c, first)
    assert allowed(first) == [["a"]] and bool((first[0, len(VOCAB):] == float("-inf")).all())
    constrain.advance(c, ids("a"))
    drafts, window = constrain.chain(c, ID["a"], ids("b", "b", "x", "b"))
    assert drafts == ids("b", "b")                                    # "x" breaks the grammar: it and its row go
    rows = torch.zeros(3, WIDTH)
    constrain.mask(c, rows, window)
    assert allowed(rows) == [["b"], ["<eos>", "b"], ["<eos>", "b"]]   # after "a": b; after "ab": b or the end
    constrain.advance(c, ids("b", "b", "<eos>"))                     # the round keeps both drafts, then ends
    assert c.finished


def test_a_follower_ranks_copy_masks_the_same():
    spec = Spec("choice", json.dumps(["yes", "no"]))
    lead, follower = constraint(spec), constrain.follow(Rank(), pack(constraint(spec)))
    for c in (lead, follower):
        constrain.advance(c, ids("y"))
    a, b = (constrain.chain(c, ID["y"], ids("e", "s", "<eos>")) for c in (lead, follower))
    assert a[0] == b[0] == ids("e", "s")                              # the stop token ends the chain: not verified
    la, lb = torch.zeros(3, WIDTH), torch.zeros(3, WIDTH)
    constrain.mask(lead, la, a[1])
    constrain.mask(follower, lb, b[1])
    assert torch.equal(la, lb) and allowed(la) == [["e"], ["s"], ["<eos>"]]


def test_thinking_rows_are_free_until_the_think_end():
    c = constraint(Spec("regex", "ab"), think_end=END)
    drafts, window = constrain.chain(c, ID["x"], ids("y", "</think>", "a", "x"))
    assert drafts == ids("y", "</think>", "a")                        # free while thinking; then the grammar's
    rows = torch.zeros(4, WIDTH)
    constrain.mask(c, rows, window)
    free = [t for t in VOCAB]
    assert allowed(rows) == [free, free, ["a"], ["b"]]
    constrain.advance(c, ids("y", "</think>", "a", "b"))
    assert not c.finished
    constrain.advance(c, ids("<eos>"))
    assert c.finished


def test_logits_from_inference_mode_are_masked_outside_it():
    c = constraint(Spec("regex", "ab+"))
    with torch.inference_mode():                   # the model's forward makes the logits here
        head, rows = torch.zeros(1, WIDTH), torch.zeros(2, WIDTH)
    with torch.no_grad():                          # the rounds mask them here
        constrain.first(c, head)
        constrain.advance(c, ids("a"))
        _, window = constrain.chain(c, ID["a"], ids("b"))
        constrain.mask(c, rows, window)
    assert allowed(head) == [["a"]] and allowed(rows) == [["b"], ["<eos>", "b"]]


def test_ended_is_true_once_the_grammar_has_taken_its_stop_token():
    assert constrain.ended(None) is False
    c = constraint(Spec("choice", json.dumps(["yes", "no"])))
    constrain.advance(c, ids("y", "e", "s"))
    assert constrain.ended(c) is False                                # the value is whole: the stop token is next
    constrain.advance(c, ids("<eos>"))
    assert constrain.ended(c) is True


def first_allowed(logits, positions, sampling):                      # the engine's sampler: the lowest allowed token
    return [next(i for i, v in enumerate(row) if v > float("-inf")) for row in logits.tolist()]


def forward(sc, window, position, taps):                             # a model that gives every row the same logits
    return torch.zeros(len(window), WIDTH), (torch.zeros(len(window), 4) if taps else None)


def test_the_serial_loop_ends_at_the_grammars_stop_token_when_eos_is_off():
    c = constraint(Spec("choice", json.dumps(["yes", "no"])))
    head = torch.zeros(1, WIDTH)                                      # the first token, as ``_run`` makes it
    constrain.first(c, head)
    first = first_allowed(head, [0], None)[0]
    constrain.advance(c, [first])
    sent = []
    this = SimpleNamespace(runner=SimpleNamespace(forward=forward), _sample=first_allowed)
    out = engine.DsEngine._serial(this, SimpleNamespace(length=7), first, 30, None, (), sent.extend, c)
    assert out == ids("y", "e", "s", "<eos>") and sent == out         # ignore_eos: eos is (), the grammar ends it


def test_the_speculative_loop_ends_at_the_grammars_stop_token_when_eos_is_off(monkeypatch):
    monkeypatch.setattr(engine, "ADAPTIVE", False)
    c = constraint(Spec("choice", json.dumps(["yes", "no"])))
    head = torch.zeros(1, WIDTH)
    constrain.first(c, head)
    first = first_allowed(head, [0], None)[0]
    constrain.advance(c, [first])
    proposed = ids("e", "s", "<eos>")                                 # the grammar cuts the stop token from the drafts
    this = SimpleNamespace(
        model=None, runner=SimpleNamespace(forward=forward), _sample=first_allowed, drafts=3,
        drafter=SimpleNamespace(absorb=lambda dc, sc, taps, position: None),
        _draft_graph=lambda sc, dc, tok, position: SimpleNamespace(run=lambda tok, position: (proposed, [0.0] * 3)))
    sent = []
    out, stats = engine.DsEngine._spec(this, SimpleNamespace(length=7), None, first, 30, None, (), sent.extend, c)
    assert out == ids("y", "e", "s", "<eos>") and sent == out
    assert stats == (1, 2, 2)                                         # one round: two drafts kept, the stop token apart


def test_a_parallel_stream_whose_grammar_has_ended_is_done_with_eos_off():
    c = constraint(Spec("choice", json.dumps(["yes", "no"])))
    s = Stream([1], 30, stop_eos=False, constraint=c)
    constrain.advance(c, ids("y", "e", "s"))
    s.take(ids("y", "e", "s"), ())
    multi.MultiDecoder._grammar_end(s)
    assert not s.done and s.finished == 0.0                           # the stop token is still to come
    constrain.advance(c, ids("<eos>"))
    s.take(ids("<eos>"), ())
    assert not s.done                                                 # ignore_eos: ``take`` does not end it
    multi.MultiDecoder._grammar_end(s)
    assert s.done and s.finished > 0.0
    stamp = s.finished
    multi.MultiDecoder._grammar_end(s)
    assert s.finished == stamp                                        # an ended stream keeps its end time


def test_a_parallel_stream_whose_grammar_failed_sends_no_token_of_that_round():
    c = constraint(Spec("choice", json.dumps(["yes", "no"])))
    sent = []
    s = Stream([1], 30, stop_eos=False, constraint=c, emit=lambda toks: sent.extend(toks))
    multi.MultiDecoder._emit(s, ids("y"), ())
    assert sent == ids("y") and not s.done
    with pytest.raises(multi.GrammarError) as failed:
        constrain.advance(c, ids("x"))                                # a token the grammar rejects
    multi.MultiDecoder._grammar_failed(s, failed.value)
    multi.MultiDecoder._emit(s, ids("x"), ())
    assert sent == ids("y")                                           # the unmasked token is not sent
    assert s.done and s.finished > 0.0 and s.error is failed.value and s.out == ids("y")
