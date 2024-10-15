"""Logging and Profiling"""

_VERBOSITY_LEVELS_FROM_STRINGS = {"error": 0, "warn": 1, "info": 2, "hint": 3}


def info(*args, **kwargs):
    return msg(*args, v="info", **kwargs)


def error(*args, **kwargs):
    args = ("Error:",) + args
    return msg(*args, v="error", **kwargs)


def warn(*args, **kwargs):
    args = ("WARNING:",) + args
    return msg(*args, v="warn", **kwargs)


def hint(*args, **kwargs):
    return msg(*args, v="hint", **kwargs)


def msg(
    *msg,
    v=None,
    time=False,
    memory=False,
    reset=False,
    end="\n",
    no_indent=False,
    t=None,
    m=None,
    r=None,
):
    r"""
    Write message to logging output.
    Log output defaults to standard output but can be set to a file
    by setting `sc.settings.log_file = 'mylogfile.txt'`.
    v : {'error', 'warn', 'info', 'hint'} or int, (default: 4)
        0/'error', 1/'warn', 2/'info', 3/'hint', 4, 5, 6...
    time, t : bool, optional (default: False)
        Print timing information; restart the clock.
    memory, m : bool, optional (default: Faulse)
        Print memory information.
    reset, r : bool, optional (default: False)
        Reset timing and memory measurement. Is automatically reset
        when passing one of ``time`` or ``memory``.
    end : str (default: '\n')
        Same meaning as in builtin ``print()`` function.
    no_indent : bool (default: False)
        Do not indent for ``v >= 4``.
    """  # noqa: D205
    # variable shortcuts
    if t is not None:
        time = t
    if m is not None:
        memory = m
    if r is not None:
        reset = r
    if v is None:
        v = 4
    if isinstance(v, str):
        v = _VERBOSITY_LEVELS_FROM_STRINGS[v]
    if v == 3:  # insert "--> " before hints
        msg = ("-->",) + msg
    if v >= 4 and not no_indent:
        msg = ("   ",) + msg
    if _settings_verbosity_greater_or_equal_than(v):
        if not time and not memory and len(msg) > 0:
            _write_log(*msg, end=end)
        if reset:
            try:
                settings._previous_memory_usage, _ = get_memory_usage()
            except:
                pass
            settings._previous_time = get_time()
        if time:
            elapsed = get_passed_time()
            msg = msg + (f"({_sec_to_str(elapsed)})",)
            _write_log(*msg, end=end)
        if memory:
            _write_log(get_memory_usage(), end=end)
