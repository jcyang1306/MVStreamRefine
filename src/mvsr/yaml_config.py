"""Typed YAML sections for `configs/*.yaml` (frozen dataclass per section, exact key set)."""

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin, get_type_hints


class ConfigError(ValueError):
    pass


def typed_value(value: Any, tp: Any, where: str) -> Any:
    """Check one YAML value against its annotation (YAML 1.1 reads e.g. `3.5e6` as a string)."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        if value is None and type(None) in get_args(tp):
            return None
        (inner,) = [a for a in get_args(tp) if a is not type(None)]
        return typed_value(value, inner, where)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list")
        (it,) = get_args(tp)
        return [typed_value(v, it, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        kt, vt = get_args(tp)
        return {
            typed_value(k, kt, where): typed_value(v, vt, f"{where}.{k}") for k, v in value.items()
        }
    if tp is bool:
        ok = isinstance(value, bool)
    elif tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif tp is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    elif tp is str:
        ok = isinstance(value, str)
    else:
        raise ConfigError(f"{where}: unsupported annotation {tp!r}")
    if not ok:
        raise ConfigError(f"{where}: expected {getattr(tp, '__name__', tp)}, got {value!r}")
    return value


def load_section(raw: dict[str, Any], cls: type[Any], name: str) -> Any:
    """`cls(**raw[name])` after requiring exactly the dataclass fields with the declared types."""
    data = raw.get(name)
    if not isinstance(data, dict):
        raise ConfigError(f"missing section '{name}'")
    names = set(cls.__dataclass_fields__)
    unknown, missing = set(data) - names, names - set(data)
    if unknown or missing:
        raise ConfigError(f"section '{name}': unknown {sorted(unknown)}, missing {sorted(missing)}")
    hints = get_type_hints(cls)
    return cls(**{k: typed_value(v, hints[k], f"{name}.{k}") for k, v in data.items()})
