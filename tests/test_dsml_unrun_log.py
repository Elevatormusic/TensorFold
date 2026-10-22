"""A tool-call block that is removed from a reply or kept as text, and does not run, is logged and counted: once for each
request, streamed or not, with the request id and no reply text. The reply itself is the one of the cut parity tests.

The replies come through the real HTTP handler with the fake engine of ``test_dsml_cut_parity`` (one character for each
token, ``max_tokens`` is the cut)."""

import re
from types import SimpleNamespace

import pytest

from tensorfold.dsml_calls import reset_suppressed, suppressed_counts
from tensorfold.server import metrics
from tests.test_dsml_cut_parity import (CLOSE, COMPLETE_THEN_OPEN, D, ONE, OPEN, READ_A, TOOLS, TWO_INVOKES, WRITE_B,
                                        Replies, block, invoke, param, served)

ID = re.compile(r"req-[0-9a-f]{12}")
SECRET = "SECRET-ARGUMENT-VALUE"


@pytest.fixture
def replies(tmp_path):
    reset_suppressed()
    yield Replies(tmp_path)
    reset_suppressed()


def request(cut, **body):
    return {"messages": [{"role": "user", "content": "Go."}], "tools": TOOLS, "max_tokens": cut, "temperature": 0,
            **body}


def lines(capsys):
    """(the WARNING lines, the done lines) that the server printed since the last call."""

    out = capsys.readouterr().out.splitlines()
    return [l for l in out if "WARNING" in l], [l for l in out if "] done req-" in l]


def run(replies, port, reply, cut, mode, **body):
    replies.engine.reply = reply
    send = replies.whole if mode == "whole" else replies.streamed
    return send(port, request(cut, **body))


MODES = ["whole", "streamed"]
OPEN_IN_FIRST_INVOKE = TWO_INVOKES.index(READ_A) + len(f'<{D} invoke name="read">\n<{D} parameter name="path" string="true">a.')
OPEN_IN_SECOND_INVOKE = TWO_INVOKES.index(WRITE_B) + len(f'<{D} invoke name="write">\n<{D} parameter name="path" st')


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cut", [OPEN_IN_FIRST_INVOKE, OPEN_IN_SECOND_INVOKE], ids=["first_invoke", "second_invoke"])
def test_an_open_block_is_logged_and_counted_once_and_the_reply_stays_the_same(replies, capsys, cut, mode, parallel):
    with served(replies) as port:
        capsys.readouterr()
        reply = run(replies, port, TWO_INVOKES, cut, mode, parallel_tool_calls=parallel)
    warnings, done = lines(capsys)
    assert reply["calls"] == [] and reply["content"] == "Checking." and reply["finish"] == "length"
    assert len(warnings) == 1, warnings
    assert "a tool call block was removed from the reply and did not run" in warnings[0]
    assert "reason=open_dsml_block" in warnings[0]
    assert f"length={cut - TWO_INVOKES.index(OPEN)} characters, at character {TWO_INVOKES.index(OPEN)}" in warnings[0]
    assert "kept" not in warnings[0]
    assert ID.search(warnings[0]).group() == ID.search(done[0]).group()        # one id for the done line and the warning
    assert suppressed_counts() == {"open_dsml_block": 1}
    assert "a.py" not in warnings[0] and "｜" not in warnings[0]


def test_streamed_and_whole_replies_each_have_their_own_line_and_count(replies, capsys):
    with served(replies) as port:
        capsys.readouterr()
        whole, streamed = replies.ask(port, TWO_INVOKES, OPEN_IN_SECOND_INVOKE)
    warnings, done = lines(capsys)
    assert whole == streamed
    assert len(warnings) == 2 and len(done) == 2
    assert {ID.search(w).group() for w in warnings} == {ID.search(d).group() for d in done}
    assert len({ID.search(w).group() for w in warnings}) == 2
    assert suppressed_counts() == {"open_dsml_block": 2}
    assert f'{metrics.PREFIX}tool_calls_suppressed_total{{reason="open_dsml_block"}} 2' in metrics.render(
        SimpleNamespace())


@pytest.mark.parametrize("mode", MODES)
def test_a_complete_block_before_an_open_block_keeps_its_call_and_the_open_block_is_logged(replies, capsys, mode):
    cut = len(COMPLETE_THEN_OPEN) - 3
    with served(replies) as port:
        capsys.readouterr()
        reply = run(replies, port, COMPLETE_THEN_OPEN, cut, mode)
    warnings, _ = lines(capsys)
    assert reply["calls"] == [("read", {"path": "a.py"})]
    assert reply["content"] == "First.\n\n\nThen."
    assert len(warnings) == 1 and "reason=open_dsml_block" in warnings[0]
    assert suppressed_counts() == {"open_dsml_block": 1}


@pytest.mark.parametrize("mode", MODES)
def test_an_open_block_that_holds_a_complete_call_says_so(replies, capsys, mode):
    nested = '<tool_call>{"name": "write", "arguments": {"path": "x"}}</tool_call>'
    reply_text = f'Go.\n{OPEN}\n<{D} invoke name="read">\n<{D} parameter name="path" string="true">{nested}'
    with served(replies) as port:
        capsys.readouterr()
        reply = run(replies, port, reply_text, len(reply_text), mode)
    warnings, _ = lines(capsys)
    assert (reply["content"], reply["calls"]) == ("Go.", [])
    assert len(warnings) == 1
    assert "reason=open_dsml_block" in warnings[0] and "it holds a complete call" in warnings[0]
    assert suppressed_counts() == {"open_dsml_block": 1}


@pytest.mark.parametrize("mode", MODES)
def test_a_block_that_is_not_well_formed_is_logged_as_kept_text(replies, capsys, mode):
    reply_text = "Go.\n" + block(invoke("read", param("path", SECRET)), "junk\n")
    with served(replies) as port:
        capsys.readouterr()
        run(replies, port, reply_text, len(reply_text) + 1, mode)
    warnings, _ = lines(capsys)
    assert len(warnings) == 1
    assert "was kept in the reply as text and did not run" in warnings[0]
    assert "reason=ill_formed_dsml_block" in warnings[0] and "at character 4" in warnings[0]
    assert SECRET not in warnings[0] and "junk" not in warnings[0]
    assert suppressed_counts() == {"ill_formed_dsml_block": 1}


@pytest.mark.parametrize("mode", MODES)
def test_a_call_to_a_tool_that_is_not_offered_is_logged(replies, capsys, mode):
    reply_text = "Go.\n" + block(invoke("apply_patch", param("path", SECRET)))
    with served(replies) as port:
        capsys.readouterr()
        run(replies, port, reply_text, len(reply_text) + 1, mode)
    warnings, _ = lines(capsys)
    assert len(warnings) == 1 and "reason=unoffered_tool_call" in warnings[0]
    assert SECRET not in warnings[0]
    assert suppressed_counts() == {"unoffered_tool_call": 1}


@pytest.mark.parametrize("mode", MODES)
def test_with_one_call_allowed_a_second_call_is_logged(replies, capsys, mode):
    with served(replies) as port:
        capsys.readouterr()
        reply = run(replies, port, TWO_INVOKES, len(TWO_INVOKES) + 1, mode, parallel_tool_calls=False)
    warnings, _ = lines(capsys)
    assert [name for name, _ in reply["calls"]] == ["read"]
    assert len(warnings) == 1 and "reason=extra_call_removed" in warnings[0]
    assert suppressed_counts() == {"extra_call_removed": 1}


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("mode", MODES)
def test_a_complete_reply_prints_no_warning_and_counts_nothing(replies, capsys, mode, parallel):
    with served(replies) as port:
        capsys.readouterr()
        reply = run(replies, port, ONE, len(ONE) + 1, mode, parallel_tool_calls=parallel)
    warnings, done = lines(capsys)
    assert reply["calls"] == [("read", {"path": "a.py"})] and reply["finish"] == "tool_calls"
    assert warnings == [] and len(done) == 1
    assert suppressed_counts() == {}
    assert CLOSE in ONE
