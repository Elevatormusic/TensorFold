"""Native and legacy image prompts keep cached IDs and whole-input fallback."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tensorfold.server.prompt_pieces import PromptPieces
from test_deepseek_v41_native_vision import NATIVE, make_app, messages


@pytest.mark.parametrize("mode", ["native", None], ids=["native", "legacy"])
@pytest.mark.parametrize("cached", [True, False], ids=["cached", "fallback"])
def test_image_templates_keep_token_cache_and_fallback(tmp_path, monkeypatch, mode, cached):
    native = mode == "native"
    template = NATIVE if native else "{% for m in messages %}{{ '<｜User｜>' }}{{ m.content }}{% endfor %}"
    app = make_app(tmp_path, monkeypatch, mode=mode, template=template)
    if not cached:
        app.tok.enable_truncation(max_length=1024)
    request = messages(n=2)
    split = app.engine.vision.split
    text_messages, pictures = split(request, native=True) if native else split(request)
    image_options = {"allow_images": True} if native else {}
    text = app.template.render(text_messages, tools=None, enable_thinking=False, **image_options)
    expected = app.engine.vision.expand(app.tok.encode(text, add_special_tokens=False).ids, pictures).token_ids
    rendered, encoded, piece_calls = [], [], []
    render, encode, ids = app.template.render, app._encode, PromptPieces.ids

    def record_render(*args, **kwargs):
        rendered.append(kwargs)
        return render(*args, **kwargs)

    def record_encode(*args, **kwargs):
        encoded.append(args)
        return encode(*args, **kwargs)

    def record_ids(self, value):
        piece_calls.append(value)
        return ids(self, value)

    monkeypatch.setattr(app.template, "render", record_render)
    monkeypatch.setattr(app, "_encode", record_encode)
    monkeypatch.setattr(PromptPieces, "ids", record_ids)
    for _ in range(2):
        prepared = app._prepare({"messages": request}, True)
        assert prepared.prompt == expected
        assert len(prepared.vision.spans) == 2
    assert len(rendered) == 2
    assert all(kwargs.get("allow_images", False) is native for kwargs in rendered)
    if cached:
        assert piece_calls == [text, text]
        assert encoded == []
        pieces = app._prompt_pieces()
        assert pieces is not None and pieces._kept
    else:
        assert piece_calls == []
        assert encoded == [(text, False), (text, False)]
        assert app._prompt_pieces() is None
