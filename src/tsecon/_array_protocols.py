# SPDX-License-Identifier: MIT
"""Guard direct-array delegation without consuming unknown containers."""

from types import BuiltinFunctionType, FunctionType
from typing import Any

import numpy as np


def supported_array_types(types: tuple[type, ...], series_type: type) -> bool:
    """Defer foreign overrides, including subclasses with their own semantics."""
    return all(t is series_type or t is np.ndarray for t in types)


def _safe_arguments(values: tuple[Any, ...]) -> bool:
    pending = [(value, False) for value in values]
    active: set[int] = set()
    checked: set[int] = set()
    while pending:
        value, leaving = pending.pop()
        kind = type(value)
        if kind in (list, tuple, dict):
            identity = id(value)
            if leaving:
                active.remove(identity)
                checked.add(identity)
                continue
            if identity in active:
                return False
            if identity in checked:
                continue
            active.add(identity)
            pending.append((value, True))
            if kind is dict:
                pending.extend((v, False) for v in value)
                pending.extend((v, False) for v in value.values())
            else:
                pending.extend((v, False) for v in value)
        elif kind is slice:
            pending.extend((v, False) for v in (value.start, value.stop, value.step))
        elif kind is np.ndarray or isinstance(value, np.generic):
            if value.dtype.hasobject:
                return False
        elif not (
            value is None
            or value is Ellipsis
            or kind
            in (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                type,
                FunctionType,
                BuiltinFunctionType,
            )
            or isinstance(value, np.dtype)
        ):
            return False
    return True


def array_function_fallback(
    func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], series_type: type
) -> Any:
    """Unwrap direct series only; reject residual operands before redispatch."""
    raw_args = tuple(np.asarray(a) if type(a) is series_type else a for a in args)
    raw_kwargs = {k: np.asarray(v) if type(v) is series_type else v for k, v in kwargs.items()}
    if not _safe_arguments(raw_args + tuple(raw_kwargs.values())):
        return NotImplemented
    return func(*raw_args, **raw_kwargs)
