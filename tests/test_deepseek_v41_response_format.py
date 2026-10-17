"""DeepSeek-V4.1 response_format: json_object and json_schema also reach the model as DeepSeek's encoding writes them
(the template's "## Response Format" block); the request's grammar still goes to the engine, which enforces it."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.cuda.app import DsApp

CHAT = {"messages": [{"role": "user", "content": "Reply in JSON."}]}
SCHEMA = {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "object"}}}


def test_plain_and_unhinted_requests_pass_unchanged():
    assert DsApp._response_format(CHAT, True) is CHAT
    for extra in ({"response_format": {"type": "text"}}, {"guided_regex": "[0-9]+"}, {"guided_choice": ["a", "b"]}):
        body = {**CHAT, **extra}
        assert DsApp._response_format(body, True) is body
    text = {"prompt": "x", "response_format": {"type": "json_object"}}
    assert DsApp._response_format(text, False) is text                    # a text completion has no template


@pytest.mark.parametrize("rf", [{"type": "json_object"}, SCHEMA])
def test_json_formats_keep_the_grammar_and_add_the_template_block(rf):
    body = {**CHAT, "response_format": rf, "chat_template_kwargs": {"x": 1}}
    out = DsApp._response_format(body, True)
    assert out["response_format"] is rf                                    # the engine still enforces it
    assert out["chat_template_kwargs"] == {"x": 1, "response_format": rf}
    assert body["chat_template_kwargs"] == {"x": 1}                        # the request is not changed in place


def test_a_clients_own_template_input_stays():
    body = {**CHAT, "response_format": SCHEMA, "chat_template_kwargs": {"response_format": {"mine": True}}}
    assert DsApp._response_format(body, True)["chat_template_kwargs"] == {"response_format": {"mine": True}}
