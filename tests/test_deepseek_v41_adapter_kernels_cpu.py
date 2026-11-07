"""The adapter delta kernels' logic, run by Triton's CPU interpreter in a process of its own (TRITON_INTERPRET is read
when a kernel is defined, so it cannot be set inside a test session that also holds compiled kernels). The same checks
run compiled on a GPU in tests/cuda/test_deepseek_v41_adapter_kernels.py."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"


def test_the_kernels_match_the_definition_and_keep_every_row_independent_under_the_cpu_interpreter():
    code = ("import json, sys; import adapter_kernel_checks as c; "
            "print(json.dumps(c.run_all('cpu')))")
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": os.pathsep.join([str(SRC), str(TESTS)])}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=900,
                          cwd=str(TESTS), check=False)
    assert done.returncode == 0, done.stderr[-3000:] + done.stdout[-1000:]
    errors = json.loads(done.stdout.strip().splitlines()[-1])
    assert set(errors) >= {"definition_bfloat16", "definition_float16", "definition_float32", "definition_wide_rank"}
    assert max(errors.values()) < 1e-3


@pytest.mark.parametrize("scale", [0.0, 1e-50])
def test_zero_stored_scales_do_not_run_overflowing_products_or_change_output_bits(scale):
    code = f"""import torch
from tensorfold.families.deepseek_v41.cuda.lora_tables import Tables
from tensorfold.families.deepseek_v41.cuda.lora_kernels import lowrank_add
tab = Tables.empty(2, 1, 128, 128, False, torch.bfloat16, 'cpu')
tab.put(1, torch.full((1, 128), 1e38), torch.ones((128, 1)), {scale!r})
out = torch.full((1, 128), -0.0)
bits = out.view(torch.int32).clone()
lowrank_add(out, torch.ones((1, 128), dtype=torch.bfloat16), torch.ones(1, dtype=torch.int64), tab)
assert int(tab.ranks[1]) == 0
assert torch.isfinite(out).all() and torch.equal(out.view(torch.int32), bits), out
"""
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": os.pathsep.join([str(SRC), str(TESTS)])}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
                          cwd=str(TESTS), check=False)
    assert done.returncode == 0, done.stdout + done.stderr
