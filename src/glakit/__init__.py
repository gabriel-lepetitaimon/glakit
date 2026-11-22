"""GLAkit Python utilities package."""

__all__ = []


def if_none[T](value: T | None, default: T) -> T:
    """Return `value` if it is not None, otherwise return `default`."""
    return value if value is not None else default
