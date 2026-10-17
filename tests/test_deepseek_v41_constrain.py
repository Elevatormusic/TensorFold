"""Structured output on the DeepSeek-V4.1 engine (families/deepseek_v41/cuda/constrain.py): a reply's grammar over a
DSpark draft chain cuts the drafts it cannot keep, masks the verify rows and follows the kept tokens, and a follower
rank's copy (rank 0's ``pack``, compiled there) masks bit for bit the same (real xgrammar, a tiny vocabulary)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
xgr = pytest.importorskip("xgrammar")

from tensorfold.engine.grammar import Grammars, Spec, pack
from tensorfold.families.deepseek_v41.cuda import constrain

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
