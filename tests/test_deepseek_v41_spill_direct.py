"""DeepSeek-V4.1 spill files written past the page cache (spill.write_safetensors, O_DIRECT): safetensors reads them
back byte for byte, for every dtype the spill stores, across staging-buffer boundaries, with and without O_DIRECT."""

import os

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")
from safetensors import safe_open  # noqa: E402

from tensorfold.families.deepseek_v41.cuda import spill  # noqa: E402


def _tensors():
    g = torch.Generator().manual_seed(7)
    out = {
        "comp.0.0": torch.randint(0, 256, (1031, 8), generator=g, dtype=torch.uint8),
        "index_k.1.1": torch.randint(0, 256, (17, 1), generator=g, dtype=torch.uint8),
        "tokens": torch.randint(0, 2**40, (777,), generator=g, dtype=torch.int64),
        "host": torch.randint(-5, 2**30, (777,), generator=g, dtype=torch.int32),
        "ring": torch.randn((3, 5, 16), generator=g).to(torch.bfloat16),
        "raw": torch.randn((4, 16), generator=g),
        "half": torch.randn((9,), generator=g).to(torch.float16),
        "mask": torch.rand((13,), generator=g) > 0.5,
        "empty": torch.zeros((0, 4), dtype=torch.float32),
    }
    if hasattr(torch, "float8_e4m3fn"):
        out["f8"] = torch.randn((33,), generator=g).to(torch.float8_e4m3fn)
    return out


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("buf", [4096, 8192, 16 << 20])
def test_round_trip(tmp_path, monkeypatch, direct, buf):
    monkeypatch.setattr(spill, "SPILL_DIRECT", direct)
    monkeypatch.setattr(spill, "_BUF", buf)
    ts = _tensors()
    path = tmp_path / "seg-run-000001-0.st"
    spill.write_safetensors(path, ts, {"p0": "0", "p1": "777", "engine": "x"})
    got = st.load_file(str(path))
    assert set(got) == set(ts)
    for k in ts:
        _same(got[k], ts[k])
    with safe_open(str(path), framework="pt") as f:
        assert f.metadata() == {"p0": "0", "p1": "777", "engine": "x"}
    # the exact size: no O_DIRECT padding left at the end
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    assert len(raw) == 8 + n + sum(t.numel() * t.element_size() for t in ts.values())


def test_offsets_aligned_to_element_size(tmp_path):
    ts = _tensors()
    path = tmp_path / "snap-run-000001-0.st"
    spill.write_safetensors(path, ts, {})
    import json
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    assert (8 + n) % 8 == 0
    header = json.loads(raw[8:8 + n])
    for k, v in header.items():
        if k != "__metadata__":
            assert v["data_offsets"][0] % ts[k].element_size() == 0, k


def test_unknown_dtype_refused(tmp_path):
    with pytest.raises(TypeError):
        spill.write_safetensors(tmp_path / "x.st", {"c": torch.zeros(2, dtype=torch.complex64)}, {})


def _open_fds():
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_failed_buffer_leaves_no_fd(tmp_path, monkeypatch):
    import mmap

    def refuse(*a, **k):
        raise OSError(12, "Cannot allocate memory")

    monkeypatch.setattr(mmap, "mmap", refuse)
    before = _open_fds()
    for _ in range(4):
        with pytest.raises(OSError):
            spill.write_safetensors(tmp_path / "x.st", _tensors(), {})
    assert _open_fds() == before


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_failed_write_closes_fd(tmp_path, monkeypatch):
    def refuse(fd, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(spill, "_write_all", refuse)
    before = _open_fds()
    for _ in range(4):
        with pytest.raises(OSError):
            spill.write_safetensors(tmp_path / "x.st", _tensors(), {})
    assert _open_fds() == before
