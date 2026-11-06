"""CPU checks for the combined adapter and reduction startup and prefill paths."""

import ast
from contextlib import nullcontext
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from test_dsv41_reduced_cpu import Link, Rank, model_comm  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("enabled", [False, True], ids=["rs0", "rs1"])
@pytest.mark.parametrize("words", [None, (11, 22, 33, 44)], ids=["stock", "adapters"])
@pytest.mark.parametrize("difference", [None, "rs", 0, 1, 2, 3], ids=["same", "rs", "word0", "word1", "word2", "word3"])
def test_nine_startup_words_agree_before_real_warm_call(monkeypatch, enabled, words, difference):
    path = ROOT / "src/tensorfold/families/deepseek_v41/cuda/engine.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DsEngine")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    packet = next(n for n in init.body if isinstance(n, ast.If) and ast.unparse(n.test) == "nccl is not None")
    warm = next(n for n in init.body if isinstance(n, ast.If) and ast.unparse(n.test) == "WARM")
    assert init.body.index(packet) < init.body.index(warm)
    compare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_same_on_every_rank")
    wrapper = ast.parse("def start(self, nccl, parallel, world):\n    pass\n")
    wrapper.body[0].body = [packet, warm]
    markov = ModuleType("tensorfold.families.deepseek_v41.cuda.markov")
    markov.ON, markov.SPLIT = True, False
    monkeypatch.setitem(sys.modules, markov.__name__, markov)
    link, errors, warmed, buffers = Link(), [], [], [[], []]

    class Flags(Rank):
        def barrier(self):
            self.link.barrier.wait()

    def start(rank):
        held = list(words or (0, 0, 0, 0))
        if rank == 1 and isinstance(difference, int):
            held[difference] += 1
        adapters = NS(words=lambda: held) if words is not None or rank == 1 and isinstance(difference, int) else None

        def empty(shape, **kwargs):
            buffers[rank].append(shape)
            return torch.empty(shape, dtype=kwargs["dtype"])

        namespace = {"__package__": "tensorfold.families.deepseek_v41.cuda", "WARM": True,
                     "RS": not enabled if rank == 1 and difference == "rs" else enabled,
                     "torch": NS(int64=torch.int64, empty=empty,
                                 tensor=lambda values, **kwargs: torch.tensor(values, dtype=kwargs["dtype"]))}
        try:
            exec(compile(ast.Module(body=[compare], type_ignores=[]), str(path), "exec"), namespace)
            exec(compile(wrapper, str(path), "exec"), namespace)
            namespace["start"](NS(vcfg=None, adapters=adapters, warm=lambda: warmed.append(rank)),
                               Flags(link, rank), 4, 2)
        except BaseException as error:
            errors.append(error)
            link.barrier.abort()

    threads = [threading.Thread(target=start, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    assert buffers == [[(18,)], [(18,)]]
    assert all(row.dtype == torch.int64 and row.numel() == 9 for row in link.sends)
    assert link.sends[0].tolist() == [0, 4, 1, 0, int(enabled), *(words or (0, 0, 0, 0))]
    if difference is None:
        assert errors == [] and sorted(warmed) == [0, 1]
        assert torch.equal(link.sends[0], link.sends[1])
    else:
        assert warmed == [] and len(errors) == 2
        assert all(isinstance(error, ValueError) and "TF_DS_RS" in str(error)
                   and "--adapter files, names, strengths and order" in str(error) for error in errors)


@pytest.mark.parametrize("enabled", [False, True], ids=["rs0", "rs1"])
@pytest.mark.parametrize("slot", [0, 1, 2], ids=["stock", "alpha", "beta"])
@pytest.mark.parametrize("rows", [3, 4], ids=["odd", "even"])
def test_actual_prefill_adds_each_adapter_before_both_rank_reductions(model_comm, enabled, slot, rows):
    path = ROOT / "src/tensorfold/families/deepseek_v41/cuda/model.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_k")
    model_comm.RS = enabled
    link, errors, posts, traces, outputs = Link(), [], [[], []], [[], []], [None, None]
    base = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
    parts = [[base + rank, -base + rank + 0.25] for rank in range(2)]
    deltas = [slot * 0.5, slot * -0.75]

    def run(rank):
        try:
            trace = traces[rank]
            comm = model_comm.Comm.__new__(model_comm.Comm)
            comm.nccl, comm.world, comm._buf = Rank(link, rank), 2, {}
            reduced, gather = comm.reduced, comm.gather

            def reduce(partial):
                trace.append("reduce")
                assert partial.dtype == torch.float32
                return reduced(partial)

            def gathered(partial):
                trace.append("gather")
                return gather(partial)

            comm.reduced, comm.gather = reduce, gathered
            bound = NS(slot=slot, rows=rows)

            def uniform(selected, count):
                assert (selected, count) == (slot, rows)
                trace.append("bind")
                return bound

            def partial(which, adapter):
                assert adapter is (bound if slot else None)
                trace.append("attention" if which == 0 else "moe")
                return parts[rank][which] + deltas[which]

            def post(value, *args):
                trace.append("post")
                posts[rank].append(value.clone())

            kernels = NS(HC_BLOCKS=1, on=lambda name: False, hc_pre=lambda *args: None, hc_post=post,
                         collapse_norm=lambda h, *args: h[:, 0].float())
            namespace = {"torch": torch, "F32": torch.float32, "BF16": torch.bfloat16, "K": kernels,
                         "_T": lambda name: nullcontext(), "mm": lambda head, x, dtype: x.to(dtype)}
            exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), namespace)
            layer = NS(hc_attn=(None, None, None), hc_ffn=(None, None, None), attn_norm=None,
                       ffn_norm=None, comp_wkv=None, idx_wq_b=None)
            model = NS(cfg=NS(dim=8, hc=1, eps=1e-6, hc_eps=1e-6, hc_iters=1),
                       w=NS(layers=[layer], norm=None, head=None), comm=comm, lora=NS(uniform=uniform),
                       attention_k=lambda *args, adapter=None, **kwargs: partial(0, adapter),
                       moe=lambda *args, adapter=None, **kwargs: partial(1, adapter))
            cache = NS(length=0)
            outputs[rank] = namespace["_forward_k"](model, cache, torch.arange(rows), 0, True, None,
                torch.zeros((rows, 1, 8), dtype=torch.bfloat16), torch.zeros((rows, 1)), None, None, adapter=slot)
            assert cache.length == rows
        except BaseException as error:
            errors.append(error)
            link.barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    fallback = ["gather"] if not enabled or rows % 2 else []
    expected_trace = ((["bind"] if slot else []) + ["attention", "reduce"] + fallback + ["post", "moe", "reduce"]
                      + fallback + ["post", "gather"])
    assert traces == [expected_trace, expected_trace]
    assert all(output.shape == (rows, 16) for output in outputs)
    for which in range(2):
        pair = [parts[rank][which] + deltas[which] for rank in range(2)]
        expected = ((pair[0] + pair[1]).to(torch.bfloat16)[None] if enabled and rows % 2 == 0
                    else torch.stack(pair))
        for rank in range(2):
            actual = posts[rank][which]
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
