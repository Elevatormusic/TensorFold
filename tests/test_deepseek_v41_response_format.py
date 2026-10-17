"""DeepSeek-V4.1 structured output as DeepSeek's API serves it: response_format json_object becomes the encoding's
"## Response Format" block in the system turn (no grammar); a schema, regex, choice or grammar is HTTP 400, since the
V4.1 engine has no constrained decoding."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.cuda.app import DsApp
from tensorfold.server.errors import RequestError

CHAT = {"messages": [{"role": "user", "content": "Reply in JSON."}]}


def test_plain_requests_pass_unchanged():
    assert DsApp._response_format(CHAT, True) is CHAT
    text = {**CHAT, "response_format": {"type": "text"}}
    assert DsApp._response_format(text, True) is text


def test_json_object_becomes_the_template_block():
    body = {**CHAT, "response_format": {"type": "json_object"}, "chat_template_kwargs": {"x": 1}}
    out = DsApp._response_format(body, True)
    assert "response_format" not in out and out["messages"] is CHAT["messages"]
    assert out["chat_template_kwargs"] == {"x": 1, "response_format": {"type": "json_object"}}
    assert body["chat_template_kwargs"] == {"x": 1}                       # the request is not changed in place


@pytest.mark.parametrize("field", [
    {"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "object"}}}},
    {"guided_regex": "[0-9]+"}, {"guided_choice": ["a", "b"]}, {"structured_outputs": {"json": {"type": "object"}}},
])
def test_enforced_formats_are_a_clear_400(field):
    with pytest.raises(RequestError, match="no constrained decoding"):
        DsApp._response_format({**CHAT, **field}, True)


def test_json_object_needs_a_chat():
    with pytest.raises(RequestError):
        DsApp._response_format({"prompt": "x", "response_format": {"type": "json_object"}}, False)
