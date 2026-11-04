"""CPU checks of decode stop decisions. These checks do not test NCCL or GPU idle state."""

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.cuda import engine as ds


class Gather:
    """Block both simulated ranks until each has reached the same collective."""

    def __init__(self):
        self.values = [None, None]
        self.barrier = threading.Barrier(2, timeout=5)
        self.rounds = [0, 0]

    def rank(self, rank):
        def gather(mine, every):
            self.values[rank] = mine.clone()
            self.barrier.wait()
            every.copy_(torch.cat(self.values))
            self.rounds[rank] += 1
            self.barrier.wait()
        return NS(all_gather=gather)


@pytest.fixture
def cpu_tensors(monkeypatch):
    def tensor(values, **kwargs):
        return torch.tensor(values, dtype=kwargs.get("dtype"))

    def empty(shape, **kwargs):
        return torch.empty(shape, dtype=kwargs.get("dtype"))

    monkeypatch.setattr(ds, "torch", NS(tensor=tensor, empty=empty, int64=torch.int64, long=torch.long, cat=torch.cat))
    monkeypatch.setattr(ds, "ADAPTIVE", False)


def make_engine(rank, link, world=2, execution="graph"):
    engine = ds.DsEngine.__new__(ds.DsEngine)
    engine.rank, engine.world = rank, world
    engine.nccl = link.rank(rank) if link else None
    engine.calls = 0

    def forward(sc, window, pos, taps):
        engine.calls += 1
        sc.length = pos + len(window)
        return [t + 1 for t in window], torch.zeros((len(window), 1))

    def eager(sc, tokens, pos, *, taps=None, **kwargs):
        logits, rows = forward(sc, tokens.tolist(), pos, taps is not None)
        if taps is not None:
            taps.append(rows)
        return logits

    engine.runner = NS(forward=forward) if execution == "graph" else None
    engine.model = NS(forward=eager)
    engine.drafts = 2
    engine.drafter = NS(absorb=lambda *args: None, draft=lambda dc, sc, tok, pos: ([tok + 1, tok + 2], []))
    engine._draft_graph = lambda sc, dc, tok, pos: NS(run=lambda *args: ([tok + 1, tok + 2], []))
    engine._sample = lambda logits, positions, sampling: logits
    return engine


def decode(engine, method, callback, eos=()):
    cache = NS(length=4)
    args = (cache, 10, 10, None, eos, callback)
    if method == "_spec":
        result, _ = engine._spec(cache, None, *args[1:])
        return result
    return engine._serial(*args)


@pytest.mark.parametrize("method", ["_serial", "_spec"])
@pytest.mark.parametrize("stop_round", [1, 3, None])
@pytest.mark.parametrize("execution", ["graph", "eager"])
def test_ranks_stop_at_the_same_callback_boundary_and_accept_next_request(cpu_tensors, method, stop_round, execution):
    link = Gather()
    engines = [make_engine(rank, link, execution=execution) for rank in range(2)]
    emitted = []

    def callback(tokens):
        emitted.append(list(tokens))
        return len(emitted) == stop_round

    def worker_callback(tokens):
        raise AssertionError("only rank 0 can call the request callback")

    with ThreadPoolExecutor(2) as pool:
        results = [pool.submit(decode, engine, method, cb) for engine, cb in
                   zip(engines, [callback, worker_callback])]
        outputs = [result.result(timeout=10) for result in results]
        assert outputs[0] == outputs[1] == sum(emitted, [])
        assert engines[0].calls == engines[1].calls == len(emitted) - 1
        assert link.rounds == [len(emitted)] * 2
        if stop_round is not None:
            assert len(emitted) == stop_round
        else:
            assert len(outputs[0]) == 10
        results = [pool.submit(decode, engine, method, cb) for engine, cb in
                   zip(engines, [lambda tokens: True, worker_callback])]
        assert [result.result(timeout=10) for result in results] == [[10], [10]]


@pytest.mark.parametrize("method", ["_serial", "_spec"])
def test_callback_error_stops_workers_before_it_is_reported(cpu_tensors, method):
    link = Gather()
    engines = [make_engine(rank, link) for rank in range(2)]

    def fail(tokens):
        raise RuntimeError("callback failed")

    with ThreadPoolExecutor(2) as pool:
        leader = pool.submit(decode, engines[0], method, fail)
        worker = pool.submit(decode, engines[1], method, lambda tokens: None)
        with pytest.raises(RuntimeError, match="callback failed"):
            leader.result(timeout=10)
        assert worker.result(timeout=10) == [10]
    assert link.rounds == [1, 1]
    assert all(engine.calls == 0 for engine in engines)


@pytest.mark.parametrize("method", ["_serial", "_spec"])
def test_single_rank_stops_without_a_collective(cpu_tensors, method):
    engine = make_engine(0, None, world=1)
    assert decode(engine, method, lambda tokens: True) == [10]
    assert engine.calls == 0


@pytest.mark.parametrize("method", ["_serial", "_spec"])
def test_eos_is_shared_before_both_ranks_exit(cpu_tensors, method):
    link = Gather()
    with ThreadPoolExecutor(2) as pool:
        results = [pool.submit(decode, make_engine(rank, link), method, lambda tokens: False, (10,))
                   for rank in range(2)]
        assert [result.result(timeout=10) for result in results] == [[10], [10]]
    assert link.rounds == [1, 1]
