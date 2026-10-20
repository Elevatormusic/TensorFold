"""--vision-max-images reaches DeepSeek-V4.1's own image path. DsApp sends image requests through DsVision.split,
not through the shared image limits, so DsVision's count check must use the server's value, not its default of 8."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.server import App
from tensorfold.families.deepseek_v41.cuda import vision as ds_vision
from tensorfold.families.deepseek_v41.cuda.app import DsApp
from tensorfold.server.errors import RequestError


def _app(monkeypatch, vision, **kwargs):
    def init(self, engine, *args, **kw):                 # App.__init__ reads the model directory: not needed here
        self.engine = engine
        self.template = SimpleNamespace(render=lambda *a, **k: "")

    monkeypatch.setattr(App, "__init__", init)
    return DsApp(SimpleNamespace(vision=vision), "model", "served", **kwargs)


def _images(n):
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,x"}}] * n}]


@pytest.fixture
def vision(monkeypatch):
    monkeypatch.setattr(ds_vision, "load_image_bytes", lambda url, allow, limit: b"x")
    monkeypatch.setattr(ds_vision, "decode", lambda data, cfg: object())
    return ds_vision.DsVision(None)


def test_the_server_limit_replaces_the_default(monkeypatch, vision):
    _app(monkeypatch, vision, vision_max_images=32)
    assert vision.max_images == 32
    _, pictures = vision.split(_images(32))
    assert len(pictures) == 32
    with pytest.raises(ValueError, match="at most 32 images"):
        vision.split(_images(33))


def test_without_the_option_the_default_stays(monkeypatch, vision):
    _app(monkeypatch, vision)
    assert vision.max_images == 8
    _app(monkeypatch, vision, vision_max_images=None)
    assert vision.max_images == 8


def test_a_text_only_engine_ignores_the_option(monkeypatch):
    app = _app(monkeypatch, None, vision_max_images=32)
    assert app.engine.vision is None


def test_the_limit_error_is_a_400(monkeypatch, vision):
    app = _app(monkeypatch, vision, vision_max_images=2)
    with pytest.raises(RequestError, match="at most 2 images"):
        app._prepare({"messages": _images(3)}, True)
