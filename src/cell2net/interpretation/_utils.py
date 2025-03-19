from collections.abc import Sequence


def is_sequence_of_ints(obj):
    return isinstance(obj, Sequence) and all(isinstance(i, int) for i in obj)


def is_sequence_of_strings(obj):
    return isinstance(obj, Sequence) and all(isinstance(i, str) for i in obj)
