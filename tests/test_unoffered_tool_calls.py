"""A call to a tool the request did not offer: by default it stays in the content as text. With
TF_DROP_UNOFFERED_TOOL_CALLS=1 the server removes it from the content and logs the tool name. Offered calls and
malformed blocks do not change."""

from __future__ import annotations

import pytest

from tensorfold.cuda.reply_text import parse_tool_calls
from tensorfold.server.tools import parse_tool_calls_from_content

D = "｜DSML｜"
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {}}}}]


def _dsml(*names: str) -> str:
    invokes = "".join(f'<{D} invoke name="{n}">\n<{D} parameter name="p" string="true">x</{D} parameter>\n'
                      f"</{D} invoke>\n" for n in names)
    return f"<{D} calls>\n{invokes}</{D} calls>"


@pytest.fixture
def drop(monkeypatch):
    monkeypatch.setenv("TF_DROP_UNOFFERED_TOOL_CALLS", "1")


def test_by_default_an_unoffered_call_stays_text(monkeypatch):
    monkeypatch.delenv("TF_DROP_UNOFFERED_TOOL_CALLS", raising=False)
    content, calls = parse_tool_calls("Done." + _dsml("apply_patch"), TOOLS)
    assert calls is None
    assert content.startswith("Done.") and "apply_patch" in content


def test_the_flag_drops_an_unoffered_call_and_logs_it(drop, capsys):
    content, calls = parse_tool_calls("Done." + _dsml("apply_patch"), TOOLS)
    assert (content, calls) == ("Done.", None)
    assert "apply_patch" in capsys.readouterr().out


def test_the_flag_keeps_the_offered_call_next_to_an_unoffered_one(drop):
    content, calls = parse_tool_calls(_dsml("apply_patch", "read"), TOOLS)
    assert content == ""
    assert [c["function"]["name"] for c in calls] == ["read"]


def test_the_flag_keeps_a_malformed_block(drop):
    text = "<tool_call>not a call</tool_call>"
    assert parse_tool_calls(text, TOOLS) == (text, None)


@pytest.mark.parametrize("value", ["0", "", "true"])
def test_only_1_turns_the_flag_on(monkeypatch, value):
    monkeypatch.setenv("TF_DROP_UNOFFERED_TOOL_CALLS", value)
    content, _ = parse_tool_calls("Done." + _dsml("apply_patch"), TOOLS)
    assert "apply_patch" in content


def test_the_mac_server_parser_follows_the_flag(monkeypatch):
    text = 'Done.<tool_call>{"name": "apply_patch", "arguments": {}}</tool_call>'
    monkeypatch.delenv("TF_DROP_UNOFFERED_TOOL_CALLS", raising=False)
    assert parse_tool_calls_from_content(text, TOOLS) == (text, None)
    monkeypatch.setenv("TF_DROP_UNOFFERED_TOOL_CALLS", "1")
    assert parse_tool_calls_from_content(text, TOOLS) == ("Done.", None)
