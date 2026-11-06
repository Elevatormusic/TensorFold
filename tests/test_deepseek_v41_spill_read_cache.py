"""Spill read cache advice follows copied reads. Advice failure cannot change restored bytes."""

import os

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from test_deepseek_v41_spill import mods, _decoder, _prompt, _rows, _serve


@pytest.mark.parametrize("fail_advice", [False, True])
def test_restored_rows_are_exact_with_and_without_cache_advice(mods, monkeypatch, fail_advice):
    multi, spill, root = mods
    md1, _ = _decoder(multi, spill, root)
    prompt = _prompt(20000, 40)
    _serve(multi, md1, prompt)
    kept, = md1.kept.values()
    expected = _rows(md1.pool, kept.base, kept.top)
    advised = []

    def advise(fd, offset, count, mode):
        advised.append(os.readlink(f"/proc/self/fd/{fd}"))
        assert (offset, count, mode) == (0, 0, os.POSIX_FADV_DONTNEED)
        if fail_advice:
            raise OSError("advice unavailable")

    monkeypatch.setattr(spill.os, "posix_fadvise", advise)
    md2, _ = _decoder(multi, spill, root)
    s = multi.Stream(prompt + [77], 1, None, draft=False, stop_eos=False)
    s.emit = lambda tokens: None
    md2.admit(s)
    assert s.cached == kept.top
    assert all(torch.equal(x, y) for x, y in zip(expected, _rows(md2.pool, s.base, kept.top)))
    files = {seg["file"] for seg in kept.disk["segments"]}
    files |= {ref["file"] for ref in kept.disk["snaps"].values()}
    assert {os.path.basename(path) for path in advised} == files


def test_advice_runs_after_the_mapping_is_closed_and_copy_is_complete(mods, monkeypatch, tmp_path):
    _, spill, _ = mods
    import safetensors
    path = tmp_path / "seg-test-0.st"
    expected = torch.arange(16)
    st.save_file({"tokens": expected}, str(path))
    safe_open, events = safetensors.safe_open, []

    class Reader:
        def __enter__(self):
            self.reader = safe_open(str(path), framework="pt")
            return self.reader.__enter__()

        def __exit__(self, *args):
            self.reader.__exit__(*args)
            del self.reader
            events.append("closed")

    def advise(*args):
        assert events == ["copied", "closed"]
        assert torch.equal(copied, expected)
        events.append("advised")

    monkeypatch.setattr(safetensors, "safe_open", lambda *args, **kwargs: Reader())
    monkeypatch.setattr(spill.os, "posix_fadvise", advise)
    with spill._read_safetensors(path) as reader:
        copied = reader.get_tensor("tokens").to("cpu", copy=True)
        events.append("copied")
    assert events == ["copied", "closed", "advised"]


def test_read_error_is_not_hidden_by_cache_advice_failure(mods, monkeypatch, tmp_path):
    _, spill, _ = mods
    path = tmp_path / "bad.st"
    path.write_bytes(b"invalid")

    def fail(*args):
        raise OSError("advice unavailable")

    monkeypatch.setattr(spill.os, "posix_fadvise", fail)
    with pytest.raises(Exception, match="header"):
        with spill._read_safetensors(path):
            pytest.fail("an invalid spill file must fail")


@pytest.mark.parametrize("replacement", ["fifo", "symlink", "file"])
def test_advice_does_not_follow_a_replaced_file(mods, monkeypatch, tmp_path, replacement):
    _, spill, _ = mods
    path, target = tmp_path / "read.st", tmp_path / "other.st"
    expected = torch.arange(8)
    st.save_file({"tokens": expected}, str(path))
    st.save_file({"tokens": expected + 1}, str(target))

    def refuse_advice(*args):
        pytest.fail("cache advice must use the original regular file only")

    monkeypatch.setattr(spill.os, "posix_fadvise", refuse_advice)
    with spill._read_safetensors(path) as reader:
        copied = reader.get_tensor("tokens").to("cpu", copy=True)
        path.unlink()
        if replacement == "fifo":
            os.mkfifo(path)
        elif replacement == "symlink":
            path.symlink_to(target)
        else:
            path.write_bytes(target.read_bytes())
    assert torch.equal(copied, expected)
