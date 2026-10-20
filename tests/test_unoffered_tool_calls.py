"""A call to a tool the request did not offer: by default it stays in the content as text. With
TF_DROP_UNOFFERED_TOOL_CALLS=1 the CUDA server removes it from the content and logs it. Offered calls and blocks that
do not parse do not change. A request with no tools drops each tool-call block."""

from __future__ import annotations

import pytest

from tensorfold.cuda.reply_text import answer_without_tools, parse_tool_calls

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


def test_the_flag_keeps_a_call_whose_arguments_do_not_parse(drop):
    text = '<tool_call>{"name": "apply_patch", "arguments": "not json"}</tool_call>'
    assert parse_tool_calls("Done." + text, TOOLS) == ("Done." + text, None)


@pytest.mark.parametrize("payload", ['{"name": "apply_patch", "arguments": []}', '{"name": "apply_patch", "arguments": 1}',
                                     '{"name": "apply_patch", "arguments": "true"}', '{"name": 5, "arguments": {}}'])
def test_the_flag_keeps_a_call_that_is_not_a_valid_call(drop, payload):
    text = f"<tool_call>{payload}</tool_call>"
    assert parse_tool_calls("Done." + text, TOOLS) == ("Done." + text, None)

def test_without_tools_the_answer_is_unchanged_by_default(monkeypatch):
    monkeypatch.delenv("TF_DROP_UNOFFERED_TOOL_CALLS", raising=False)
    answer = "Done." + _dsml("apply_patch")
    assert answer_without_tools(answer, finished=True) == answer


def test_without_tools_the_flag_drops_each_call_and_logs_it(drop, capsys):
    answer = "Done." + _dsml("apply_patch") + " Bye."
    assert answer_without_tools(answer, finished=True) == "Done. Bye."
    assert "offered no tools" in capsys.readouterr().out


def test_without_tools_a_streamed_call_never_shows(drop, capsys):
    answer = "Done." + _dsml("apply_patch")
    for end in range(len(answer) + 1):
        assert answer_without_tools(answer[:end], finished=False) == answer[:min(end, 5)]
    assert capsys.readouterr().out == ""                  # the log comes with the finished answer only
