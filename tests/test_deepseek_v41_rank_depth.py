"""The "rank" depth policy (families/deepseek_v41/cuda/multi.py ``_rank_k``): the drafts a round verifies across all
streams are the allocation with the most expected tokens a millisecond under the cost table, each stream's a
prefix, within the round's row cap; one stream gets the k of the "conf" policy."""

from __future__ import annotations

import itertools
import math
import random

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.cuda import multi


def decoder(max_rows: int = 16, depth: int = 5) -> multi.MultiDecoder:
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.max_rows, dec.depth_most = max_rows, depth
    return dec


def value(confs, ks, live):
    tokens = float(live)
    for cf, k in zip(confs, ks):
        surv = 1.0
        for j in range(k):
            surv *= 1.0 / (1.0 + math.exp(-cf[j]))
            tokens += surv
    rows = live + sum(ks)
    return tokens / (multi.ROUND_MS[min(len(multi.ROUND_MS), rows) - 1] + multi.DRAFT_MS)


def test_rank_is_the_best_allocation():
    rng = random.Random(5)
    dec = decoder()
    for _ in range(200):
        n = rng.randint(2, 4)
        confs = [[rng.uniform(-3.0, 6.0) for _ in range(5)] for _ in range(n)]
        want = [rng.randint(0, 5) for _ in range(n)]
        ks = dec._rank_k(n, confs, want)
        assert all(0 <= k <= w for k, w in zip(ks, want)) and n + sum(ks) <= dec.max_rows
        best = max(value(confs, c, n) for c in itertools.product(*[range(w + 1) for w in want])
                   if n + sum(c) <= dec.max_rows)
        assert value(confs, ks, n) == pytest.approx(best, rel=1e-12)


def test_one_stream_takes_the_conf_k():
    rng = random.Random(9)
    dec = decoder()
    for _ in range(200):
        cf = [rng.uniform(-3.0, 6.0) for _ in range(5)]
        assert dec._rank_k(1, [cf], [5]) == [dec._choose_k(cf, 5, 1)]


def test_sure_streams_fill_the_row_cap():
    dec = decoder()
    ks = dec._rank_k(4, [[9.0] * 5] * 4, [5] * 4)
    assert sum(ks) == dec.max_rows - 4 and max(ks) - min(ks) <= 1
    ks = dec._rank_k(4, [[9.0] * 5, [-4.0] * 5, [-4.0] * 5, [-4.0] * 5], [5] * 4)
    assert ks[0] == 5 and ks[1:] == [0, 0, 0]
