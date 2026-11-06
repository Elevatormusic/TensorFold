"""Check the exact communicator source with CPU collectives and no Triton import."""
from __future__ import annotations

import ast
import os
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.comm import NCCL, NCCL_SUM, _DTYPES
from tensorfold.cuda.rdma import Hybrid, RdmaGather

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_comm():
    source = ROOT / "src/tensorfold/families/deepseek_v41/cuda/model.py"
    tree = ast.parse(source.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "Comm")
    module = ModuleType("reduced_cpu_contract")
    module.__dict__.update(torch=torch, F32=torch.float32, BF16=torch.bfloat16, RS=True, os=os)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), module.__dict__)
    return module


class Link:
    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=3)
        self.sends, self.calls = [None, None], []

    def swap(self, rank, send):
        self.sends[rank] = send.clone()
        self.barrier.wait()
        both = list(self.sends)
        self.barrier.wait()
        return both


class Rank:
    def __init__(self, link, rank):
        self.link, self.rank, self.world = link, rank, 2

    def reduce_scatter(self, send, recv):
        a, b = self.link.swap(self.rank, send)
        self.link.calls.append(("reduce_scatter", send.dtype))
        recv.copy_((a + b).reshape(2, -1)[self.rank])

    def all_gather(self, send, recv):
        a, b = self.link.swap(self.rank, send)
        self.link.calls.append(("all_gather", send.dtype))
        recv.copy_(torch.cat([a, b]))


class SmallRdma:
    fits = RdmaGather.fits

    def __init__(self, nccl, max_bytes):
        self.nccl, self.max_bytes = nccl, max_bytes

    def all_gather(self, send, recv):
        self.nccl.link.calls.append(("rdma", send.dtype))
        self.nccl.all_gather(send, recv)


def both_ranks(module, p0, p1, rdma_bytes=None):
    link, outputs, errors = Link(), [None, None], []

    def run(rank, partial):
        try:
            comm = module.Comm.__new__(module.Comm)
            nccl = Rank(link, rank)
            comm.nccl = nccl if rdma_bytes is None else Hybrid(nccl, SmallRdma(nccl, rdma_bytes))
            comm.world, comm._buf = 2, {}
            outputs[rank] = comm.reduced(partial)
        except BaseException as error:
            errors.append(error)
            link.barrier.abort()

    threads = [threading.Thread(target=run, args=pair, daemon=True) for pair in ((0, p0), (1, p1))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    if errors:
        raise errors[0]
    return outputs, link.calls


@pytest.mark.parametrize("strided", [False, True])
def test_reduced_partials_keep_exact_bf16_bits(model_comm, strided):
    generator = torch.Generator().manual_seed(41)
    parts = [torch.randn((8, 128), generator=generator) for _ in range(2)]
    if strided:
        parts = [part[:, ::2] for part in parts]
        assert not parts[0].is_contiguous()
    outputs, calls = both_ranks(model_comm, *parts)
    expected = (parts[0] + parts[1]).to(torch.bfloat16)[None]
    for output in outputs:
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    assert sorted(name for name, dtype in calls) == ["all_gather", "all_gather", "reduce_scatter", "reduce_scatter"]
    assert all(dtype == torch.bfloat16 for name, dtype in calls if name == "all_gather")


def test_rounding_boundaries_cancellation_and_large_values(model_comm):
    values = torch.tensor([0.0, -0.0, 2**-126, -2**-126, 1.00390625, 1.00390625 + 2**-23,
                           2**100, -2**100, 3.4e38, -3.4e38], dtype=torch.float32)
    p0 = values.repeat(4).reshape(4, 10)
    p1 = values.flip(0).repeat(4).reshape(4, 10)
    outputs, _ = both_ranks(model_comm, p0, p1)
    expected = (p0 + p1).to(torch.bfloat16)[None]
    assert all(torch.equal(output.view(torch.int16), expected.view(torch.int16)) for output in outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_other_dtypes_use_the_unchanged_gather(model_comm, dtype):
    p0 = torch.arange(32, dtype=dtype).reshape(4, 8)
    p1 = p0 + 1
    outputs, calls = both_ranks(model_comm, p0, p1)
    assert all(torch.equal(output, torch.stack([p0, p1])) for output in outputs)
    assert [name for name, dtype in calls] == ["all_gather", "all_gather"]


@pytest.mark.parametrize("rows,enabled", [(7, True), (8, False)])
def test_odd_rows_and_disabled_mode_use_the_gather(model_comm, rows, enabled):
    model_comm.RS = enabled
    p0 = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
    p1 = p0 + 1
    outputs, calls = both_ranks(model_comm, p0, p1)
    assert all(torch.equal(output, torch.stack([p0, p1])) for output in outputs)
    assert [name for name, dtype in calls] == ["all_gather", "all_gather"]


@pytest.mark.parametrize("rdma_bytes,expected_path", [(64, "reduce_scatter"), (4096, "rdma")])
def test_real_hybrid_dispatch_respects_the_rdma_dtype_gate(model_comm, rdma_bytes, expected_path):
    p0 = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    p1 = -p0 + 1
    outputs, calls = both_ranks(model_comm, p0, p1, rdma_bytes)
    if expected_path == "rdma":
        assert all(torch.equal(output, torch.stack([p0, p1])) for output in outputs)
        assert ("rdma", torch.float32) in calls
        assert not any(name == "reduce_scatter" for name, dtype in calls)
    else:
        expected = (p0 + p1).to(torch.bfloat16)[None]
        assert all(torch.equal(output.view(torch.int16), expected.view(torch.int16)) for output in outputs)
        assert not any(name == "rdma" for name, dtype in calls)


@pytest.mark.parametrize("world", [1, 3])
def test_other_world_sizes_keep_gather_layout(model_comm, world):
    partial = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    comm = model_comm.Comm.__new__(model_comm.Comm)
    comm.world, comm._buf = world, {}

    def gather(send, recv):
        recv.copy_(torch.cat([send + rank for rank in range(world)]))

    comm.nccl = SimpleNamespace(all_gather=gather)
    assert torch.equal(comm.reduced(partial), torch.stack([partial + rank for rank in range(world)]))


def test_nccl_wrapper_uses_receive_count_and_current_stream(monkeypatch):
    calls = []
    comm = NCCL.__new__(NCCL)
    comm.world, comm.comm = 2, object()
    comm.lib = SimpleNamespace(ncclReduceScatter=lambda *args: calls.append(args) or 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=123))
    send, receive = torch.arange(12, dtype=torch.float32), torch.empty(6, dtype=torch.float32)
    comm.reduce_scatter(send, receive)
    assert calls == [(send.data_ptr(), receive.data_ptr(), 6, _DTYPES[torch.float32], NCCL_SUM, comm.comm, 123)]


@pytest.mark.parametrize("receive", [torch.empty(5), torch.empty(6, dtype=torch.bfloat16)])
def test_nccl_wrapper_rejects_bad_count_or_dtype_before_launch(receive):
    comm = NCCL.__new__(NCCL)
    comm.world = 2
    with pytest.raises(ValueError, match="send must hold world"):
        comm.reduce_scatter(torch.arange(12, dtype=torch.float32), receive)


@pytest.mark.parametrize("switches", [(False, False), (True, True), (False, True)])
def test_startup_agrees_reducer_mode_before_warmup(monkeypatch, switches):
    source = ROOT / "src/tensorfold/families/deepseek_v41/cuda/engine.py"
    tree = ast.parse(source.read_text())
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DsEngine")
    init = next(node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    packet = next(node for node in init.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "nccl is not None")
    warm = next(node for node in init.body if isinstance(node, ast.If) and ast.unparse(node.test) == "WARM")
    assert init.body.index(packet) < init.body.index(warm)
    wrapper = ast.parse("def agree(self, nccl, parallel, world):\n    pass\n")
    wrapper.body[0].body = [packet]
    compare = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_same_on_every_rank")
    markov = ModuleType("tensorfold.families.deepseek_v41.cuda.markov")
    markov.ON, markov.SPLIT = True, True
    monkeypatch.setitem(sys.modules, markov.__name__, markov)
    cpu_torch = SimpleNamespace(int64=torch.int64,
        tensor=lambda values, **kwargs: torch.tensor(values, dtype=kwargs["dtype"]),
        empty=lambda shape, **kwargs: torch.empty(shape, dtype=kwargs["dtype"]))
    link, errors, warmed = Link(), [], []

    class Flags(Rank):
        def barrier(self):
            self.link.barrier.wait()

    def start(rank):
        namespace = {"__package__": "tensorfold.families.deepseek_v41.cuda", "torch": cpu_torch,
                     "RS": switches[rank]}
        try:
            exec(compile(wrapper, str(source), "exec"), namespace)
            exec(compile(ast.Module(body=[compare], type_ignores=[]), str(source), "exec"), namespace)
            namespace["agree"](SimpleNamespace(vcfg=None, adapters=None), Flags(link, rank), 4, 2)
            warmed.append(rank)
        except BaseException as error:
            errors.append(error)
            link.barrier.abort()

    threads = [threading.Thread(target=start, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    if switches[0] == switches[1]:
        assert errors == [] and sorted(warmed) == [0, 1]
    else:
        assert warmed == [] and len(errors) == 2
        assert all(isinstance(error, ValueError) and "TF_DS_RS" in str(error) for error in errors)
