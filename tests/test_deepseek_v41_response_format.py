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


def test_the_structured_outputs_alias_goes_with_the_block():
    body = {**CHAT, "structured_outputs": {"json_object": True}}
    out = DsApp._response_format(body, True)
    assert "structured_outputs" not in out and "response_format" not in out
    assert out["chat_template_kwargs"] == {"response_format": {"type": "json_object"}}


@pytest.mark.parametrize("kwargs", [False, [], "x", 3])
def test_json_object_keeps_the_template_kwargs_check(kwargs):
    with pytest.raises(RequestError, match="chat_template_kwargs must be a JSON object"):
        DsApp._response_format({**CHAT, "response_format": {"type": "json_object"}, "chat_template_kwargs": kwargs},
                               True)


def test_null_template_kwargs_count_as_none():
    out = DsApp._response_format({**CHAT, "response_format": {"type": "json_object"}, "chat_template_kwargs": None},
                                 True)
    assert out["chat_template_kwargs"] == {"response_format": {"type": "json_object"}}


JSON_OBJECT = {**CHAT, "response_format": {"type": "json_object"}}


@pytest.mark.parametrize("field, named", [
    ({"guided_regex": "[0-9]+"}, "guided_regex"),
    ({"guided_choice": ["a", "b"]}, "guided_choice"),
    ({"guided_json": {"type": "object"}}, "guided_json"),
    ({"guided_grammar": 'root ::= "a"'}, "guided_grammar"),
    ({"structured_outputs": {"regex": "[0-9]+"}}, "structured_outputs.regex"),
    ({"structured_outputs": {"json_object": True, "choice": ["a"]}}, "structured_outputs.choice"),
    ({"structured_outputs": "regex"}, "structured_outputs"),
])
def test_json_object_with_a_competing_control_is_a_clear_400(field, named):
    with pytest.raises(RequestError, match="serves json_object only") as err:
        DsApp._response_format({**JSON_OBJECT, **field}, True)
    assert named in str(err.value)


def test_json_object_alone_and_the_alias_alone_keep_the_template_block():
    block = {"response_format": {"type": "json_object"}}
    assert DsApp._response_format(JSON_OBJECT, True)["chat_template_kwargs"] == block
    alias = {**CHAT, "structured_outputs": {"json_object": True}}
    assert DsApp._response_format(alias, True)["chat_template_kwargs"] == block


def test_unset_competing_controls_do_not_block_json_object():
    body = {**JSON_OBJECT, "guided_regex": None, "structured_outputs": {"json_object": True, "regex": None,
                                                                         "choice": False}}
    out = DsApp._response_format(body, True)
    assert out["chat_template_kwargs"] == {"response_format": {"type": "json_object"}}
    assert "structured_outputs" not in out and "response_format" not in out
