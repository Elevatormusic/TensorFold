"""DeepSeek-V4.1's image path hands the template the block list it is written for.

DsVision.split used to flatten a message's content into one string that already held the placeholder. The
checkpoint's own chat_template.jinja is written for a block list: its joined_text macro turns an image/image_url
block into the placeholder itself, and its guard raises on a string that carries the placeholder. So every image
request was refused ("Message content contains image special token") with the stock template. These tests render a
template that has the same guard, so the shape the engine hands over is what is checked, not a stub that accepts
anything.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("jinja2")

from jinja2 import Environment
from jinja2.exceptions import TemplateError

from tensorfold.cuda.server import App, PreparedRequest
from tensorfold.families.deepseek_v41.cuda import vision as ds_vision
from tensorfold.families.deepseek_v41.cuda.app import DsApp

# The two guards of the checkpoint's template that this path meets, and the image branch that emits the placeholder.
TEMPLATE = """
{%- set IMG = "<｜deepseek_image｜>" -%}
{%- macro joined_text(blocks) -%}
    {%- set jt = namespace(parts=[]) -%}
    {%- for b in blocks -%}
        {%- if b.type == "text" -%}
            {%- if IMG in (b.text or "") -%}{{- raise_exception("Text block contains image placeholder " ~ IMG) -}}{%- endif -%}
            {%- set jt.parts = jt.parts + [b.text or ""] -%}
        {%- elif b.type in ["image", "image_url"] -%}
            {%- set jt.parts = jt.parts + [IMG] -%}
        {%- endif -%}
    {%- endfor -%}
    {{- jt.parts | join("\\n\\n") -}}
{%- endmacro -%}
{%- for m in messages -%}
    {%- set content = m.get("content") -%}
    {%- if content is string and IMG in content -%}
        {{- raise_exception("Message content contains image special token " ~ IMG) -}}
    {%- endif -%}
    {%- if content is iterable and content is not string and content is not mapping -%}
        {{- joined_text(content) -}}
    {%- else -%}
        {{- content -}}
    {%- endif -%}
{%- endfor -%}
"""


def _template():
    env = Environment()
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(TemplateError(message))
    return env.from_string(TEMPLATE)


@pytest.fixture
def vision(monkeypatch):
    monkeypatch.setattr(ds_vision, "load_image_bytes", lambda url, allow, limit: b"x")
    monkeypatch.setattr(ds_vision, "decode", lambda data, cfg: object())
    return ds_vision.DsVision(None)


def _one_image():
    return [{"role": "user", "content": [{"type": "text", "text": "what colour is this?"},
                                         {"type": "image_url", "image_url": {"url": "data:,x"}}]}]


def test_split_keeps_the_block_list_the_template_reads(vision):
    out, pictures = vision.split(_one_image())
    assert len(pictures) == 1
    assert out[0]["content"] == [{"type": "text", "text": "what colour is this?"}, {"type": "image"}]


def test_the_stock_template_renders_the_placeholder(vision):
    out, _ = vision.split(_one_image())
    text = _template().render(messages=out)
    assert text.count(ds_vision.PLACEHOLDER) == 1


def test_a_flattened_string_is_what_the_guard_refuses(vision):
    """The shape split used to produce: the template refuses it, which is why the engine must not produce it."""
    out, _ = vision.split(_one_image())
    flattened = [{**out[0], "content": "what colour is this?\n\n" + ds_vision.PLACEHOLDER}]
    with pytest.raises(TemplateError, match="image special token"):
        _template().render(messages=flattened)


def test_the_engine_hands_the_template_the_block_list(monkeypatch, vision):
    """DsApp._prepare renders through the base class with allow_images, so the blocks survive normalize_messages.

    The base class is the real one: only the parts that read a checkpoint (template, tokenizer, encode) are stubbed,
    so the allow_images wiring this test is about actually runs.
    """
    seen = {}

    def render(messages, **kwargs):
        seen["messages"] = messages
        seen["allow_images"] = kwargs.get("allow_images")
        return _template().render(messages=messages)

    def init(self, engine, *args, **kw):                 # App.__init__ reads the model directory: not needed here
        self.engine = engine
        self.template = SimpleNamespace(render=render,
                                        prepared=lambda m, allow_images=False: m,
                                        efforts=frozenset())
        self.max_tokens = 8
        self.tok = None
        self.default_thinking = False
        self.late_system = "system"
        self.merge_leading = True
        self.image_limits = None

    monkeypatch.setattr(App, "__init__", init)
    app = DsApp(SimpleNamespace(vision=vision), "model", "served")
    monkeypatch.setattr(App, "_encode", lambda self, text, flag: [1])
    monkeypatch.setattr(App, "_plain_text", lambda self: None)
    monkeypatch.setattr(App, "_context_limit", lambda self: 4096)
    monkeypatch.setattr(App, "_grammars", lambda self: SimpleNamespace(compile=lambda spec: None))
    monkeypatch.setattr(DsApp, "sampling_for", lambda self, body, ids: None)
    # expand reads the checkpoint's image_token_id: this test is about the shape the template gets, not the spans
    monkeypatch.setattr(ds_vision.DsVision, "expand",
                        lambda self, ids, pictures: SimpleNamespace(token_ids=ids))

    app._prepare({"messages": _one_image()}, True)
    assert seen["allow_images"] is True
    assert seen["messages"][0]["content"] == [{"type": "text", "text": "what colour is this?"},
                                              {"type": "image"}]
