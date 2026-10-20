"""A concurrent round runs in a captured graph of the next size up, and the warm-up covers every graph it can need."""

import importlib.util
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src/tensorfold/families/deepseek_v41/cuda/graph_keys.py"
spec = importlib.util.spec_from_file_location("graph_keys_under_test", SRC)
keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keys)    # no torch: the module holds only the key arithmetic

CAP = 1_048_576 + 16 + 8         # --context 1048576 with the engine's 16 rows and 8 spare


def test_buckets_double_from_the_minimum_and_stop_at_the_window():
    assert keys.context_buckets(CAP, CAP) == [CAP, 1_048_576, 524_288, 262_144, 131_072, 65_536, 32_768, 16_384,
                                              8_192, 4_096, 2_048, 1_024]
    assert keys.context_buckets(5_000, 5_000) == [5_000, 4_096, 2_048, 1_024]


def test_the_default_sizes_end_at_the_most_rows():
    assert keys.row_sizes(16, (1, 2, 4, 8, 16)) == [1, 2, 4, 8, 16]
    assert keys.row_sizes(16, (2, 5)) == [2, 5, 16]
    assert keys.row_sizes(12, (1, 2, 4, 8, 16)) == [1, 2, 4, 8, 12]      # none above the engine most rows
    assert keys.row_sizes(16, (0,)) == list(range(1, 17))                # 0: a graph for every count
    assert keys.row_sizes(16, ()) == list(range(1, 17))


def test_a_round_takes_the_smallest_size_that_holds_it():
    sizes = keys.row_sizes(16, (1, 2, 4, 8, 16))
    assert [keys.padded_rows(r, sizes) for r in range(1, 17)] == [1, 2, 4, 4, 8, 8, 8, 8] + [16] * 8
    assert keys.padded_rows(16, sizes) == 16


def test_every_round_the_engine_can_run_has_a_warm_key():
    for sizes in (keys.row_sizes(16, (1, 2, 4, 8, 16)), keys.row_sizes(16, (0,)), keys.row_sizes(16, (3, 6, 12))):
        warm = set(keys.round_keys(CAP, CAP, sizes))
        assert len(warm) == 12 * len(sizes)
        for deepest in list(range(1, 70_000, 97)) + [2**k + d for k in range(10, 21) for d in (-1, 0, 1)] + [CAP]:
            for rows in range(1, 17):
                assert (keys.padded_rows(rows, sizes), keys.bucket_for(deepest, CAP)) in warm, (rows, deepest)


def test_the_default_warm_set_is_no_larger_than_the_one_before():
    # the graphs held at ready: 64 before (16 row counts at 4 buckets), now 5 sizes at all 12 buckets
    assert len(keys.round_keys(CAP, CAP, keys.row_sizes(16, (1, 2, 4, 8, 16)))) == 60


def test_warm_order_captures_the_largest_bucket_first():
    order = keys.round_keys(CAP, CAP, [1, 2, 4, 8, 16])
    assert order[0] == (16, CAP) and order[4] == (1, CAP)
    assert [b for _, b in order] == sorted((b for _, b in order), reverse=True)
    assert len(set(order)) == len(order)


def test_a_small_window_has_few_keys():
    assert keys.round_keys(2_000, 2_000, [1, 2, 4]) == [(r, b) for b in (2_000, 1_024) for r in (4, 2, 1)]
