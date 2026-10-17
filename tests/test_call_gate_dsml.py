"""tool_choice "required" on DeepSeek-V4.x: a DSML calls block opens with a plain "<" before the DSML token, so the gate
forces that head, the token, then the block's first invoke up to an offered name (fake tokenizer: a token a character,
the DSML mark one token)."""

from __future__ import annotations

import string

from tensorfold.engine.call_gate import DSML, CallGate, call_format, dsml_format

# V4.1 (DeepSeek's encoding/README.md): leading spaces before calls, invoke and parameter; V4: no space, tool_calls
V41_CALL = (f'<｜Assistant｜></think>\n\n<{DSML} calls>\n<{DSML} invoke name="tfprobe_fn">\n</{DSML} invoke>\n'
            f'</{DSML} calls><｜end▁of▁sentence｜>')
V4_CALL = f'</think>\n\n<{DSML}tool_calls>\n<{DSML}invoke name="tfprobe_fn">\n</{DSML}invoke>\n</{DSML}tool_calls>'
V41_LEAD = f' calls>\n<{DSML} invoke name="'

MARK, NLNL = 1, 2                                   # the DSML token, a blank token
CHARS = sorted(set(V41_CALL.replace(DSML, "") + string.ascii_letters + "_ ?"))
TEXT = {MARK: DSML, NLNL: "\n\n", **{100 + i: c for i, c in enumerate(CHARS)}}
IDS = {v: k for k, v in TEXT.items()}


def text(token):
    return TEXT[int(token)]


def encode(written):
    out, i = [], 0
    while i < len(written):
        if written.startswith(DSML, i):
            out.append(MARK)
            i += len(DSML)
        else:
            out.append(IDS[written[i]])
            i += 1
    return out


def gate(names=("get_weather", "get_time")):
    head, opener, lead, tail = dsml_format(V41_CALL, "tfprobe_fn")
    return CallGate(IDS.get(opener, MARK), {NLNL}.__contains__, text=text, encode=encode, lead=lead,
                    names=list(names), tail=tail, head=head)


def test_the_dsml_format_comes_from_the_rendered_call():
    assert dsml_format(V41_CALL, "tfprobe_fn") == ("<", DSML, V41_LEAD, '"')
    assert dsml_format(V4_CALL, "tfprobe_fn") == ("<", DSML, f'tool_calls>\n<{DSML}invoke name="', '"')
    assert dsml_format("no calls here", "tfprobe_fn") is None
    assert dsml_format(f'<{DSML} calls>', "tfprobe_fn") is None                 # no call to the probe
    assert call_format(V41_CALL, "tfprobe_fn", ["<tool_call>"]) is None         # why the old gate refused V4.1


def test_prose_becomes_the_head_the_token_and_the_lead():
    assert gate().cut(encode("Sure")) == (0, [*encode("<"), MARK, *encode(V41_LEAD)])
    assert gate().cut([NLNL, *encode("Sure")]) == (1, [*encode("<"), MARK, *encode(V41_LEAD)])   # blank passes


def test_a_sampled_call_passes_through():
    written = encode(f'<{DSML}{V41_LEAD}get_weather">\n')
    assert gate().cut([NLNL, *written]) is None


def test_a_broken_head_or_bare_token_is_fixed():
    assert gate().cut(encode("<x")) == (1, [MARK, *encode(V41_LEAD)])           # "<" then prose: the token is owed
    assert gate().cut([MARK, *encode(" calls")]) == (0, [*encode("<"), MARK, *encode(V41_LEAD)])   # no "<" first


def test_the_name_holds_to_the_offered_tools():
    lead = encode(f'<{DSML}{V41_LEAD}')
    assert gate().cut([*lead, *encode("get_x")]) == (len(lead) + 4, encode('weather"'))   # the first name "get_" starts
    assert gate().cut([*lead, *encode("zz")]) == (len(lead), encode('get_weather"'))       # a name no tool has


def test_a_forced_call_reaches_an_offered_name():
    check = gate()
    hit = check.cut(encode("Sure"))
    for token in [*hit[1], *encode('get_time"')]:
        assert not check.done
        check.observe(token)
    assert check.done and check.cut(encode("anything")) is None
    assert "".join(text(t) for t in hit[1]) == f'<{DSML} calls>\n<{DSML} invoke name="'


def test_a_head_needs_text_and_encode():
    bare = CallGate(MARK, {NLNL}.__contains__, head="<")                        # no text or encode: no head
    assert bare.head == "" and bare.cut(encode("S")) == (0, [MARK])
    assert CallGate(MARK, {NLNL}.__contains__).cut([MARK, 7]) is None           # without a head: as before
