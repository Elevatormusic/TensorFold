"""Tool communicators keep gather layout in the model's CPU-dispatched prefill."""
from __future__ import annotations

import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from test_dsv41_reduced_cpu import model_comm  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def source_class(path, name):
    return next(node for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.ClassDef) and node.name == name)


@pytest.fixture(params=["small_bench", "grouped_model_check"])
def tool_comm(request, model_comm):
    path = ROOT / "tools/dsv41" / (request.param + ".py")
    if request.param == "small_bench":
        spec = importlib.util.spec_from_file_location("small_bench_cpu_contract", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        comm = module.FakeComm()
    else:
        namespace = {"M": SimpleNamespace(Comm=model_comm.Comm), "torch": torch}
        node = source_class(path, "Repeat")
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        comm = namespace["Repeat"]()
    assert comm.world == 2 and comm.nccl is None
    return comm, model_comm


def same_bits(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@pytest.mark.parametrize("enabled", [False, True], ids=["rs0", "rs1"])
@pytest.mark.parametrize("dtype,rows,strided", [(torch.float32, 4, True), (torch.bfloat16, 3, False)],
                         ids=["even-fp32-strided", "odd-bf16"])
def test_tool_reduced_keeps_legacy_gather(tool_comm, monkeypatch, enabled, dtype, rows, strided):
    comm, module = tool_comm
    monkeypatch.setenv("TF_DS_RS", "1" if enabled else "0")
    module.RS = enabled
    partial = torch.arange(rows * 16, dtype=dtype).reshape(rows, 16)
    partial = partial[:, ::2] if strided else partial
    before = partial.clone()
    same_bits(comm.reduced(partial), torch.stack([partial, partial]))
    same_bits(partial, before)


@pytest.mark.parametrize("enabled", [False, True], ids=["rs0", "rs1"])
def test_actual_prefill_dispatch_accepts_tool_communicators(tool_comm, monkeypatch, enabled):
    comm, module = tool_comm
    monkeypatch.setenv("TF_DS_RS", "1" if enabled else "0")
    module.RS = enabled
    path = ROOT / "src/tensorfold/families/deepseek_v41/cuda/model.py"
    model = source_class(path, "Model")
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_k")
    partials = [torch.arange(32, dtype=torch.float32).reshape(4, 8),
                -torch.arange(32, dtype=torch.float32).reshape(4, 8)]
    posts = []
    kernels = SimpleNamespace(HC_BLOCKS=1, on=lambda name: False, hc_pre=lambda *args: None,
                              hc_post=lambda gathered, *args: posts.append(gathered.clone()),
                              collapse_norm=lambda h, *args: h[:, 0].float())
    namespace = {"torch": torch, "F32": torch.float32, "BF16": torch.bfloat16, "K": kernels,
                 "_T": lambda name: nullcontext(), "mm": lambda head, x, dtype: x.to(dtype)}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), namespace)
    layer = SimpleNamespace(hc_attn=(None, None, None), hc_ffn=(None, None, None), attn_norm=None,
                            ffn_norm=None, comp_wkv=None, idx_wq_b=None)
    cfg = SimpleNamespace(dim=8, hc=1, eps=1e-6, hc_eps=1e-6, hc_iters=1)
    model = SimpleNamespace(cfg=cfg, w=SimpleNamespace(layers=[layer], norm=None, head=None), comm=comm, lora=None,
                            attention_k=lambda *args, **kwargs: partials[0], moe=lambda *args, **kwargs: partials[1])
    cache = SimpleNamespace(length=0)
    output = namespace["_forward_k"](model, cache, torch.arange(4), 0, True, None,
                                       torch.zeros((4, 1, 8), dtype=torch.bfloat16),
                                       torch.zeros((4, 1)), None, None)
    assert cache.length == 4 and output.shape == (4, 16)
    assert len(posts) == 2
    for actual, partial in zip(posts, partials):
        same_bits(actual, torch.stack([partial, partial]))


def test_small_prompt_check_uses_the_same_benchmark_loader():
    path = ROOT / "tools/dsv41/small_prompt_check.py"
    tree = ast.parse(path.read_text())
    assert any(isinstance(node, ast.ImportFrom) and node.module == "small_bench"
               and any(alias.name == "load_model" for alias in node.names) for node in tree.body)
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_model"
               for node in ast.walk(tree))
    bench = ast.parse((ROOT / "tools/dsv41/small_bench.py").read_text())
    loader = next(node for node in bench.body if isinstance(node, ast.FunctionDef) and node.name == "load_model")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FakeComm"
               for node in ast.walk(loader))
