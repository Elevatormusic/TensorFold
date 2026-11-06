"""CPU checks for adapter identity across token caching, images, and prefill rotation."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")
pytest.importorskip("safetensors")

from adapter_fakes import CFG, model_dir, standard
from tensorfold.families.deepseek_v41.cuda import lora_registry as lr
from tensorfold.families.deepseek_v41.cuda import vision as ds_vision
from test_deepseek_v41_adapter_multi import KEY_A, KEY_B, decoder, mods, ref
from test_deepseek_v41_spill import CHUNK, _prompt, _rows
from test_deepseek_v41_native_vision import NATIVE, make_app, messages


def registry(tmp_path):
    return lr.AdapterRegistry.load([
        ("alpha", str(standard(tmp_path / "a", 1)), 1.0),
        ("beta", str(standard(tmp_path / "b", 2)), 1.0),
    ], CFG, model_dir(tmp_path / "base"), ("test",))


@pytest.mark.parametrize("mode", ["text", "native", "legacy"])
@pytest.mark.parametrize("cached", [True, False], ids=["cached", "fallback"])
def test_equal_prompt_ids_keep_adapter_and_image_kv_scopes_separate(mods, tmp_path, monkeypatch, mode, cached):
    multi, spill, root = mods
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    template = NATIVE if mode == "native" else "{% for m in messages %}{{ '<｜User｜>' }}{{ m.content }}{% endfor %}"
    app = make_app(app_dir, monkeypatch, mode="native" if mode == "native" else None, template=template)
    reg = registry(tmp_path)
    app.engine.adapters, app.adapter_models = reg, reg.model_map()
    if not cached:
        app.tok.enable_truncation(max_length=65536)
    pixel = [1.0]

    def picture(*args):
        return NS(tokens=3, n_vit_h=1, n_vit_w=1, n_llm_h=1, n_llm_w=3,
                  patches=torch.tensor([[pixel[0]]], dtype=torch.bfloat16))

    monkeypatch.setattr(ds_vision, "decode", picture)
    request = ([{"role": "user", "content": "x" * 9000}] if mode == "text" else messages())
    if mode != "text":
        request[0]["content"][-1]["text"] = "x" * 9000
    md, calls = decoder(multi, spill, root)
    md.e.adapter_ok = lambda held: held is None or reg.check(held)
    md.e.tower = NS(span_rows=lambda pic: torch.zeros((pic.tokens, 4)))
    md.e._share_rows = lambda rows, count: rows
    prepared_ids, keys, first_cache = [], [], None

    def complete(prepared):
        held = reg.ref(prepared.adapter) if prepared.adapter is not None else None
        s = multi.Stream(list(prepared.prompt), 1, None, draft=False, stop_eos=False,
                         adapter=held, vision=prepared.vision)
        s.emit = lambda tokens: None
        md.admit(s)
        while s in md.filling:
            md._fill(s)
        md.finish([s])
        return s

    for index, name in enumerate(("alpha", "beta", None, "alpha")):
        body = {"messages": request, **({"adapter": name} if name is not None else {})}
        prepared = app.prepare(body, True)
        assert prepared.adapter == name
        prepared_ids.append(prepared.prompt)
        if mode != "text":
            assert len(prepared.vision.spans) == 1
            assert prepared.prompt.count(app.tok.token_to_id(ds_vision.PLACEHOLDER)) == 3
        pieces = app._prompt_pieces()
        assert (pieces is not None) is cached
        if cached:
            if first_cache is None:
                first_cache = dict(pieces._kept)
                assert first_cache
            else:
                assert pieces._kept.keys() == first_cache.keys()
                assert all(pieces._kept[k] is value for k, value in first_cache.items())
        calls.clear()
        s = complete(prepared)
        keys.append(s.keys)
        expected_cut = 8192 if index == 3 else 0
        assert s.cached == expected_cut
        assert calls == [(expected_cut, 0 if name is None else reg.ref(name).slot)]
    assert all(ids == prepared_ids[0] for ids in prepared_ids)
    assert all(np.array_equal(value, keys[0]) for value in keys)
    held_rows = {kept.adapter: _rows(md.pool, kept.base, kept.top) for kept in md.kept.values()}
    assert set(held_rows) == {"", reg.ref("alpha").key, reg.ref("beta").key}
    assert any(not torch.equal(a, b) for a, b in zip(held_rows[""], held_rows[reg.ref("alpha").key]))
    assert any(not torch.equal(a, b) for a, b in zip(held_rows[reg.ref("alpha").key], held_rows[reg.ref("beta").key]))
    if mode != "text":
        pixel[0] = 2.0
        changed = app.prepare({"messages": request, "adapter": "alpha"}, True)
        assert changed.prompt == prepared_ids[0]
        calls.clear()
        s = complete(changed)
        assert not np.array_equal(s.keys, keys[0])
        assert s.cached == 0 and calls == [(0, reg.ref("alpha").slot)]


@pytest.mark.parametrize("stop", ["count", "callback"])
def test_rotating_adapter_prefills_keep_rank_order_and_cancel_scope(mods, monkeypatch, stop):
    multi, spill, root = mods
    monkeypatch.setattr(multi, "ADMIT_GIB", 0.0)
    monkeypatch.setattr(multi, "QUICK_ROWS", 0)
    decoders = [decoder(multi, spill, root / str(rank))[0] for rank in range(2)]
    for md in decoders:
        md.slack0, md.gate_log = None, 0.0
        for slot, key in ((1, KEY_A), (2, KEY_B)):
            count = 10 if slot == 1 and stop == "callback" else 1
            s = multi.Stream(_prompt(CHUNK * 3 + 1, 37), count, None, draft=False, stop_eos=False,
                             adapter=ref(multi, slot, key))
            s.emit = (lambda tokens: True) if slot == 1 and stop == "callback" else (lambda tokens: None)
            md.admit(s)
    leader, follower = decoders
    digests = [[], []]
    for rank, md in enumerate(decoders):
        md.e.rank, md.e.world = rank, 2
        md._agree = lambda label, plan, rank=rank: digests[rank].append((label, plan))
    sent = []
    leader._send = lambda op: sent.append(op)
    for turn in range(8):
        done = leader.round()
        op = sent[-1]
        assert op == ["fill", turn % 2]
        other_done = follower._fill(next(s for s in follower.filling if s.sid == op[1]))
        assert [(s.sid, s.filled, s.adapter) for s in leader.filling] == [
            (s.sid, s.filled, s.adapter) for s in follower.filling]
        assert [s.sid for s in done] == [s.sid for s in other_done]
        assert digests[0] == digests[1]
        if done:
            assert len(done) == 1 and done[0].adapter.slot == op[1] + 1
            leader.finish(done)
            follower.finish(other_done)
            if turn == 6:
                assert [s.adapter.slot for s in leader.filling] == [2]
    assert not leader.filling and not follower.filling
    assert {k.adapter for k in leader.kept.values()} == {KEY_A, KEY_B}
    assert leader._shape() == follower._shape()
    rows = {k.adapter: _rows(leader.pool, k.base, k.top) for k in leader.kept.values()}
    assert any(not torch.equal(a, b) for a, b in zip(rows[KEY_A], rows[KEY_B]))
