"""The CUDA server's reply text: streamed decoding, think blocks and tool-call parsing (the Mac lane server's rules)."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterator
from typing import Any, NamedTuple

from tensorfold.dsml_calls import (CALL_CLOSE as _CALL_CLOSE, CALL_OPEN as _CALL_OPEN, DSML_OPEN as _DSML_OPEN,
                                   EXTRA_CALL, ILL_FORMED, NO_TOOLS, OPEN_DSML, OPEN_TAG, UNOFFERED, UNRESOLVED,
                                   count_suppressed, parse_calls_block, top_level_blocks)
from tensorfold.server.stopping import StopPolicy
from tensorfold.server.tools import parse_glm_tool_call_block
from tensorfold.tool_parameters import decode_parameter, parameter_schemas

_TOOL_FUNCTION_BLOCK_RE = re.compile(r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$", re.IGNORECASE | re.DOTALL)
_TOOL_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.IGNORECASE | re.DOTALL)

# DeepSeek-V4.x DSML tool calls (the format DeepSeek's own encoding.py writes and its chat template asks for):
#   <｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n<｜DSML｜ parameter name="k" string="true">v</｜DSML｜ parameter>\n
#   </｜DSML｜ invoke>\n</｜DSML｜ calls>   (string="false": the value is JSON)
# ``tensorfold.dsml_calls`` finds the blocks: a call is a complete, well formed block at the top level of the reply.


class Unrun(NamedTuple):
    """A tool-call block that did not run: removed from the reply, or kept in it as text."""

    reason: str            # the label of tool_calls_suppressed_total
    action: str            # "removed" or "kept_as_text"
    offset: int            # character where the block begins in the reply, or -1 when the parser does not know
    length: int            # characters of the block (an open block: through the end of the reply)
    holds_call: bool = False    # an open block that holds a complete call


_WHY = {
    OPEN_DSML: "the reply ended inside a DSML calls block",
    OPEN_TAG: "a <tool_call> tag never closed and holds a complete call",
    ILL_FORMED: "the DSML calls block is not well formed",
    UNOFFERED: "the call names a tool that the request did not offer",
    UNRESOLVED: "the call has no usable name or no arguments as a JSON object",
    EXTRA_CALL: "the request allows one call and the reply made more",
    NO_TOOLS: "the request offered no tools",
}
_VERB = {"removed": "was removed from the reply", "kept_as_text": "was kept in the reply as text"}
_MAX_LINES = 5                    # lines for one reply; the rest are counted and summed in one more line


def report_unrun(events: list[Unrun], request_id: str = "") -> None:
    """Count each block that did not run and print one ``WARNING`` line for it. A line holds the request id, the
    action, the reason, the length and the place, and nothing from the reply: no text of a block, no tool name, no
    argument."""

    if not events:
        return
    count_suppressed(event.reason for event in events)
    who = request_id or "req-?"
    for event in events[:_MAX_LINES]:
        where = f", at character {event.offset}" if event.offset >= 0 else ""
        holds = ", it holds a complete call" if event.holds_call else ""
        print(f"[tensorfold] WARNING: {who} a tool call block {_VERB[event.action]} and did not run: "
              f"reason={event.reason} ({_WHY[event.reason]}), length={event.length} characters{where}{holds}",
              flush=True)
    if len(events) > _MAX_LINES:
        print(f"[tensorfold] WARNING: {who} {len(events) - _MAX_LINES} more tool call blocks did not run "
              f"in this reply", flush=True)


def _segments(text: str, suppressed: list[Unrun]) -> Iterator[str | tuple[str, str]]:
    """The reply as text (a plain string) and call blocks ((its markup, its JSON payload)), in order.

    A DSML invoke becomes the payload of a ``<tool_call>`` block. Only blocks at the top level are calls: an open
    ``<tool_call>`` block, a block that is not well formed and everything inside a block stay text or argument text.
    An open DSML calls block (the reply ended inside it) is removed with everything after it: it is not a call, and
    the stream hides it too. Each block that did not run adds an ``Unrun`` to ``suppressed``: every open DSML block,
    an open ``<tool_call>`` tag only when it holds a complete block, and every closed DSML block that is not well
    formed."""

    blocks, open_at = top_level_blocks(text)
    dsml_open = open_at >= 0 and text.startswith(_DSML_OPEN, open_at)
    if open_at >= 0:
        opener = _DSML_OPEN if dsml_open else _CALL_OPEN
        holds = bool(top_level_blocks(text[open_at + len(opener):])[0])     # a complete block that the open one holds
        if dsml_open:
            suppressed.append(Unrun(OPEN_DSML, "removed", open_at, len(text) - open_at, holds))
        elif holds:                 # a lone <tool_call> tag is text, as the model wrote it: told only with a call in it
            suppressed.append(Unrun(OPEN_TAG, "kept_as_text", open_at, len(text) - open_at, True))
    cursor = 0
    for block in blocks:
        yield text[cursor:block.start]
        cursor = block.end
        if block.kind == "tag":
            yield text[block.start:block.end], block.inner.strip()
            continue
        invokes = parse_calls_block(block.inner)
        if invokes is None:
            suppressed.append(Unrun(ILL_FORMED, "kept_as_text", block.start, block.end - block.start))
            yield text[block.start:block.end]
            continue
        for name, args in invokes:
            payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
            yield _CALL_OPEN + payload + _CALL_CLOSE, payload
    yield text[cursor:open_at] if dsml_open else text[cursor:]


def _partial_tag(text: str, tag: str, *, any_case: bool = False) -> int:
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if (text[-k:].lower() == tag[:k].lower()) if any_case else text.endswith(tag[:k]):
            return k
    return 0


class StreamDecoder:
    """Decode a shared token window to preserve leading spaces and byte boundaries, deferring incomplete characters."""

    def __init__(self, tok, skip: tuple[int, ...] = ()):
        self.tok, self.skip = tok, frozenset(skip)
        self.ids: list[int] = []
        self.text = ""
        self.prefix = 0             # window start
        self.read = 0               # tokens already reflected in ``text``

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def add(self, new: list[int]) -> str:
        self.ids.extend(t for t in new if t not in self.skip)
        before = self._decode(self.ids[self.prefix:self.read])
        after = self._decode(self.ids[self.prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            self.text += after[len(before):]
            self.prefix, self.read = self.read, len(self.ids)
        return self.text

    def final(self) -> str:
        """Everything, including a trailing partial character (as decoding it all at once gives)."""

        before = self._decode(self.ids[self.prefix:self.read])
        return self.text + self._decode(self.ids[self.prefix:])[len(before):]



class StopStrings:
    """A request's ``stop`` strings, decided token by token, so a reply cuts at the same token however its tokens arrive."""

    def __init__(self, strings: tuple[str, ...], tok, skip: tuple[int, ...] = ()):
        self.strings, self.tok, self.skip = strings, tok, frozenset(skip)
        # a new match lies in the last (its UTF-8 length) tokens, plus eight as the Mac's ``StopPolicy`` keeps
        self.tail = max((len(s.encode()) for s in strings), default=0) + 8

    def hit(self, tokens: list[int]) -> bool:
        """Whether the text through the newest token holds a stop string (earlier tokens were checked already)."""

        ids = [t for t in tokens[-self.tail:] if t not in self.skip]
        text = self.tok.decode(ids, skip_special_tokens=False)
        return any(stop in text for stop in self.strings)

    def visible(self, text: str, *, partial: bool = False) -> str:
        """The text before the first match; ``partial`` also holds back an end that may begin one (the Mac's rule)."""

        return StopPolicy.visible(self, text, partial=partial)


def drop_unoffered_calls() -> bool:
    """TF_DROP_UNOFFERED_TOOL_CALLS=1: a call to a tool that the request did not offer is removed from the content
    and logged, instead of kept as text. A model can call a tool from an earlier turn or another harness; a client
    then shows its markup as text."""

    return os.environ.get("TF_DROP_UNOFFERED_TOOL_CALLS", "0") == "1"


def keep_unoffered_call(text: str) -> str:
    """The content kept for a parsed call to a tool the request did not offer: its text, or nothing with the flag."""

    return "" if drop_unoffered_calls() else text


def answer_without_tools(answer: str, *, finished: bool, request_id: str = "") -> str:
    """The answer of a request that offers no tools: as it is, or with the flag, without its tool-call blocks (each
    call is unoffered). While streaming, a block that has begun waits until it ends, as with tools. The finished
    answer logs and counts each block that was removed."""

    if not drop_unoffered_calls():
        return answer
    # the parser finds tags in any case, and keeps a block that is not well formed as text
    shown = hide_tool_calls(answer, finished=finished, any_case=True, keep_ill_formed=True)
    if finished and shown != answer:
        blocks, open_at = top_level_blocks(answer, any_case=True)
        removed = [Unrun(NO_TOOLS, "removed", b.start, b.end - b.start) for b in blocks
                   if not (b.kind == "dsml" and parse_calls_block(b.inner) is None)]
        if open_at >= 0:
            removed.append(Unrun(NO_TOOLS, "removed", open_at, len(answer) - open_at))
        report_unrun(removed, request_id)
    return shown


def hide_tool_calls(text: str, *, finished: bool, any_case: bool = False, keep_ill_formed: bool = False) -> str:
    """The text without its top-level tool-call blocks. A block that is still open hides the rest of the text, and
    while streaming, so does a tail that may begin an opening tag. The parser finds the same blocks.

    ``keep_ill_formed``: a closed DSML block that is not well formed is not a call, so it stays in the text, as the
    parser keeps it in the content."""

    blocks, open_at = top_level_blocks(text, any_case=any_case)
    end = len(text) if open_at < 0 else open_at
    out: list[str] = []
    pos = 0
    for block in blocks:
        if block.start >= end:
            break
        if keep_ill_formed and block.kind == "dsml" and parse_calls_block(block.inner) is None:
            continue
        out.append(text[pos:block.start])
        pos = block.end
    tail = text[pos:end]
    if open_at >= 0:
        return "".join(out) + tail
    if not finished:
        tail = tail[: len(tail) - _partial_tag(tail, _CALL_OPEN, any_case=any_case)]
    shown = "".join(out) + tail
    return shown if finished else shown[: len(shown) - _partial_tag(shown, _DSML_OPEN)]


def _finite_json(value: Any) -> bool:
    """True when the value is valid strict JSON (json.loads accepts NaN and Infinity, the call path does not)."""

    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError):
        return False
    return True


def _tool_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    return str((fn or tool).get("name") or "").strip() if isinstance(tool, dict) else ""


def parse_tool_calls(text: str, tools: list[dict[str, Any]], *, max_calls: int | None = None,
                     request_id: str = "") -> tuple[str, list[dict[str, Any]] | None]:
    """Qwen ``<function=name><parameter=k>v</parameter></function>``, GLM ``name<arg_key>..`` or JSON calls."""

    if not tools:
        return text, None
    known = {_tool_name(t).lower(): _tool_name(t) for t in tools}
    schemas = parameter_schemas(tools)
    calls: list[dict[str, Any]] = []
    residue: list[str] = []
    suppressed: list[Unrun] = []
    for segment in _segments(text, suppressed):
        if isinstance(segment, str):
            residue.append(segment)
            continue
        markup, block = segment
        if max_calls is not None and len(calls) >= max_calls:
            suppressed.append(Unrun(EXTRA_CALL, "removed", -1, len(markup)))
            continue
        name, args = None, {}
        parsed = False                                  # the whole payload parsed, not only its name
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                fn = payload.get("function") if isinstance(payload.get("function"), dict) else payload
                name = fn.get("name")
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                # a call needs a name and object arguments that are strict JSON
                parsed = isinstance(name, str) and isinstance(args, dict) and _finite_json(args)
        except (json.JSONDecodeError, AttributeError):
            m = _TOOL_FUNCTION_BLOCK_RE.match(block)
            if m:
                if max_calls is not None and _TOOL_PARAMETER_BLOCK_RE.sub("", m.group(2)).strip():
                    suppressed.append(Unrun(UNRESOLVED, "removed", -1, len(markup)))
                    continue
                name = m.group(1).strip()
                # typed parameters (array, object, number...) decode per the tool's schema, as the Mac server does
                props = schemas.get(name.lower(), {})
                args = {key: decode_parameter(value, props.get(key, {}))
                        for key, value in ((p.group(1).strip(), p.group(2))
                                           for p in _TOOL_PARAMETER_BLOCK_RE.finditer(m.group(2)))}
                parsed = not _TOOL_PARAMETER_BLOCK_RE.sub("", m.group(2)).strip()   # no text outside the parameters
            else:
                glm = parse_glm_tool_call_block(block, tools, complete=max_calls is not None)
                if glm is not None:
                    name, args = glm
                    # no text outside the arguments, as the complete parser requires
                    parsed = parse_glm_tool_call_block(block, tools, complete=True) is not None
        if not name or str(name).lower() not in known:
            whole = bool(name and parsed)                # a whole call to a tool that is not offered
            if max_calls is None:
                kept = keep_unoffered_call(markup) if whole else markup
                residue.append(kept)
                action = "kept_as_text" if kept else "removed"
            else:
                action = "removed"                       # single-call mode removes the block anyway
            suppressed.append(Unrun(UNOFFERED if whole else UNRESOLVED, action, -1, len(markup)))
            continue
        if max_calls is not None:
            try:
                if not isinstance(args, dict):
                    raise ValueError("the arguments are not an object")
                json.dumps(args, allow_nan=False)
            except (ValueError, TypeError):
                suppressed.append(Unrun(UNRESOLVED, "removed", -1, len(markup)))
                continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": known[str(name).lower()],
                                   "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
    report_unrun(suppressed, request_id)
    return "".join(residue).strip(), calls or None


# -- chat template -------------------------------------------------------------------------
