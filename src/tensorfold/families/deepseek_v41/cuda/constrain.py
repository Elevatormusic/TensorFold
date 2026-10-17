"""Structured output on the DeepSeek-V4.1 engine: a reply's grammar (tensorfold.engine.grammar) over DSpark's draft
chains, the same on every rank.

Every rank samples its own copy of the gathered logits (keyed sampling: no broadcast), so every rank keeps its own copy
of the reply's grammar: rank 0's ``pack`` goes with the request, the followers compile it (``follow``). Each round the
grammar cuts the drafts no accepted path can keep, masks the verify rows, and follows the tokens the round keeps. The
ranks walk the same grammar over the same tokens, so their masks, and their samples, agree bit for bit."""

from __future__ import annotations

from typing import Any, Sequence


def follow(engine, packed: Sequence[int]):
    """The constraint rank 0 ``pack``ed, compiled on this rank from the same checkpoint, or None for []."""

    if not packed:
        return None
    from tensorfold.engine import grammar

    return grammar.compiler(engine, engine.model_dir, engine.eos).follow(list(packed))


def chain(constraint, pending: int, drafts: list[int]) -> tuple[list[int], Any]:
    """The drafts the grammar keeps after ``pending`` (each row's parent is the row before it), and the window that
    masks the verify rows; ``(drafts, None)`` without a grammar. ``pending`` is already in the grammar."""

    if constraint is None:
        return drafts, None
    window = constraint.window([pending, *drafts], list(range(-1, len(drafts))))
    return list(window.tokens[1:]), window


def mask(constraint, logits, window) -> None:
    """The verify rows' tokens the grammar does not allow at -inf (``logits`` [rows, vocab], in place)."""

    if constraint is not None:
        _in_place(constraint, logits, window)


def first(constraint, logits) -> None:
    """The prompt's last row: the reply's first token under the grammar (in place)."""

    if constraint is not None:
        _in_place(constraint, logits, None)


def _in_place(constraint, logits, window) -> None:
    # The model makes its logits under torch.inference_mode(), and the rounds run under torch.no_grad() only. An
    # in-place write to an inference tensor outside inference mode raises, so the mask runs inside it.
    import torch

    with torch.inference_mode():
        constraint.mask(logits, window)


def advance(constraint, tokens: Sequence[int]) -> None:
    """Follow the tokens a round keeps (each one chosen under the grammar's mask)."""

    if constraint is not None and tokens:
        constraint.advance(tokens)
