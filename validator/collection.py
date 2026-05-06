from collections.abc import Iterable


def to_list[T](value: T | Iterable[T]) -> list[T]:
    if isinstance(value, Iterable):
        return list(value)
    return [value]
