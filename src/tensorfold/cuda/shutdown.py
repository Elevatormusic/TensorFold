"""End the serving process on SIGTERM or Ctrl-C without interpreter finalization.

The scheduler worker is a daemon thread. If it is in a torch call (a prompt chunk) when the main thread returns,
CPython kills it at its next GIL re-acquire: pthread_exit unwinds it through a noexcept C++ frame, and the C++
runtime aborts with "terminate called without an active exception". ``leave`` waits a short time for the worker,
flushes the logs and calls os._exit, so finalization never runs beside a live worker.
"""

from __future__ import annotations

import os
import sys
from typing import Any, NoReturn

GRACE_S = 3.0                    # TF_SHUTDOWN_GRACE_S: how long the worker may take to stop (earlyoom escalates fast)

_exit = os._exit                 # a test replaces it


def grace() -> float:
    """The wait for the worker, in seconds."""

    try:
        return max(0.0, float(os.environ.get("TF_SHUTDOWN_GRACE_S") or GRACE_S))
    except ValueError:
        return GRACE_S


def leave(app: Any, code: int = 0) -> NoReturn:
    """Stop the app's scheduler worker after its current round if that ends in time, flush stdout and stderr, end the
    process with ``code``."""

    scheduler = getattr(getattr(app, "engine", None), "scheduler", None)
    if scheduler is not None:
        try:
            scheduler.close(timeout=grace(), drain=False)
        except Exception:                                # noqa: BLE001  (the process ends here either way)
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:                                # noqa: BLE001
            pass
    _exit(code)
    raise SystemExit(code)                               # (only when a test replaced _exit)
