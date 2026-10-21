"""DeepSeek-V4.1 reads a special-token spelling in message text as plain text, as deepseek-ai/deepseek-recipe does;
only the grounding tokens in an assistant's text and the server's own image placeholder stay tokens."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.server import App
from tensorfold.families.deepseek_v41.cuda import vision as ds_vision
from tensorfold.families.deepseek_v41.cuda.app import GROUNDING, DsApp


def _app(monkeypatch, vision):
    def init(self, engine, *args, **kw):                 # App.__init__ reads the model directory: not needed here
        self.engine = engine
        self.template = SimpleNamespace(render=lambda *a, **k: "")

    monkeypatch.setattr(App, "__init__", init)
    return DsApp(SimpleNamespace(vision=vision), "model", "served")


def test_the_family_asks_for_plain_message_text():
    assert DsApp.plain_message_text is True
    assert set(GROUNDING) == {f"<｜{tag}｜>" for tag in ("ref", "/ref", "box", "/box", "point", "/point", "polygon",
                                                        "/polygon")}
    assert DsApp.plain_assistant_keep == GROUNDING


def test_the_image_placeholder_is_kept_only_where_the_server_writes_it(monkeypatch):
    assert _app(monkeypatch, None).plain_content_keep == ()          # no vision: the spelling is plain text
    assert _app(monkeypatch, ds_vision.DsVision(None)).plain_content_keep == (ds_vision.PLACEHOLDER,)
