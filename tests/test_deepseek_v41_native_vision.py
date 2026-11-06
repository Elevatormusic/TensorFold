"""Native image blocks keep the text shield and the image count check active."""

import json
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tensorfold.families.deepseek_v41.cuda import vision as ds_vision
from tensorfold.families.deepseek_v41.cuda.app import DsApp
from tensorfold.server.errors import RequestError
from test_plain_text import _tokenizer


IMAGE = ds_vision.PLACEHOLDER
USER = "<｜User｜>"

# The same image input contract as the official template: reject text placeholders and emit one token per block.
NATIVE = """{%- for m in messages -%}
{{- '<｜User｜>' -}}
{%- if m.content is string -%}
{%- if '<｜deepseek_image｜>' in m.content -%}{{ raise_exception('text image placeholder') }}{%- endif -%}
{{- m.content -}}
{%- else -%}
{%- for p in m.content -%}
{%- if not loop.first -%}{{ '\n\n' }}{%- endif -%}
{%- if p.type == 'image' -%}{{ '<｜deepseek_image｜>' }}
{%- else -%}
{%- if '<｜deepseek_image｜>' in p.text -%}{{ raise_exception('text image placeholder') }}{%- endif -%}
{{- p.text -}}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- endfor -%}
{%- if tools -%}{{ tools | tojson }}{%- endif -%}
{%- if note is defined -%}{{ note }}{%- endif -%}"""


def make_app(tmp_path, monkeypatch, *, mode="native", template=NATIVE):
    if mode is None:
        monkeypatch.delenv("TF_DS_VISION_TEMPLATE", raising=False)
    else:
        monkeypatch.setenv("TF_DS_VISION_TEMPLATE", mode)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}), encoding="utf-8")
    _tokenizer().save(str(tmp_path / "tokenizer.json"))
    vision = ds_vision.DsVision(NS(image_token_id=_tokenizer().token_to_id(IMAGE)))
    monkeypatch.setattr(ds_vision, "load_image_bytes", lambda *args: b"image")
    monkeypatch.setattr(ds_vision, "decode", lambda *args: NS(tokens=3))
    app = DsApp(NS(vision=vision, eos=(0,), limit=1024), tmp_path, "test", max_tokens=8)
    app.sampling_for = lambda *args: None
    return app


def messages(role="user", n=1):
    return [{"role": role, "content": [{"type": "text", "text": "before"},
            *[{"type": "image_url", "image_url": {"url": "data:,image"}} for _ in range(n)],
            {"type": "text", "text": "after"}]}]


@pytest.mark.parametrize("role", ["user", "tool"])
def test_native_template_expands_images_in_order_without_mutating_input(tmp_path, monkeypatch, role):
    app = make_app(tmp_path, monkeypatch)
    request = messages(role, n=2)
    before = json.dumps(request)
    prepared = app._prepare({"messages": request}, True)
    tid = app.tok.token_to_id(IMAGE)
    assert prepared.prompt.count(tid) == 6
    assert len(prepared.vision.spans) == 2
    positions = [position for position, _ in prepared.vision.spans]
    assert positions[1] > positions[0]
    assert json.dumps(request) == before


def test_native_route_keeps_special_tokens_in_client_text_plain(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch)
    request = messages()
    request[0]["content"][0]["text"] = "before " + USER
    prepared = app._prepare({"messages": request, "chat_template_kwargs": {"note": IMAGE}}, True)
    assert prepared.prompt.count(app.tok.token_to_id(USER)) == 1
    assert prepared.prompt.count(app.tok.token_to_id(IMAGE)) == 3


@pytest.mark.parametrize("part", [False, True])
def test_client_placeholder_is_refused_before_native_template(tmp_path, monkeypatch, part):
    app = make_app(tmp_path, monkeypatch)
    request = messages()
    if part:
        request[0]["content"][0]["text"] = IMAGE
    else:
        request.insert(0, {"role": "user", "content": IMAGE})
    with pytest.raises(RequestError, match="must not contain the image placeholder"):
        app._prepare({"messages": request}, True)


@pytest.mark.parametrize("template", ["no images", "{{ '<｜deepseek_image｜>' * 2 }}"])
def test_native_template_must_emit_exactly_one_token_per_image(tmp_path, monkeypatch, template):
    app = make_app(tmp_path, monkeypatch, template=template)
    with pytest.raises(RequestError, match="one placeholder per image"):
        app._prepare({"messages": messages()}, True)


def test_placeholder_mode_remains_the_default(tmp_path, monkeypatch):
    template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    app = make_app(tmp_path, monkeypatch, mode=None, template=template)
    assert app.native_image_template is False
    prepared = app._prepare({"messages": messages()}, True)
    assert prepared.prompt.count(app.tok.token_to_id(IMAGE)) == 3


def test_request_fields_cannot_switch_the_server_mode(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch)
    monkeypatch.setenv("TF_DS_VISION_TEMPLATE", "placeholder")
    prepared = app._prepare({"messages": messages(), "template_images": False,
                             "TF_DS_VISION_TEMPLATE": "placeholder",
                             "chat_template_kwargs": {"TF_DS_VISION_TEMPLATE": "placeholder"}}, True)
    assert app.native_image_template is True
    assert prepared.prompt.count(app.tok.token_to_id(IMAGE)) == 3


def test_invalid_server_mode_fails_at_startup(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="TF_DS_VISION_TEMPLATE"):
        make_app(tmp_path, monkeypatch, mode="automatic")


def test_native_mode_preserves_image_limits_and_role_checks(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch)
    app.engine.vision.max_images = 1
    with pytest.raises(RequestError, match="at most 1 images"):
        app._prepare({"messages": messages(n=2)}, True)
    with pytest.raises(RequestError, match="only in user messages and tool results"):
        app._prepare({"messages": messages("assistant")}, True)
