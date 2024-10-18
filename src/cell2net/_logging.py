"""Logging and Profiling"""

from __future__ import annotations

_VERBOSITY_LEVELS_FROM_STRINGS = {"error": 0, "warn": 1, "info": 2, "hint": 3}


def info(*args, **kwargs):
    return msg(*args, v="info", **kwargs)


def msg(
    *msg,
    v=None,
):

    return None


def _write_log(*msg, end="\n"):
    """Write message to log output, ignoring the verbosity level.
    This is the most basic function.

    Parameters
    ----------
    *msg :
        One or more arguments to be formatted as string. Same behavior as print
        function.
    """

    # print(*msg, end=end)
