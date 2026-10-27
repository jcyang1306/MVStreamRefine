"""Generic JSON / NPZ (de)serialization of contract dataclasses (DATA_CONTRACTS.md §Serialization).

- JSON: UTF-8, sorted keys, `schema_version` + `type` in every top-level object, matrices as
  row-major nested lists, floats with `repr` precision, NaN/inf rejected (`allow_nan=False`).
- NPZ: uncompressed; every ndarray becomes one entry keyed by its dotted field path, all
  other fields go to a `__meta__` entry holding the same JSON document with
  `{"__npz__": "<key>"}` placeholders. Loading never unpickles.
Decoding is driven by the dataclass type annotations, so dtypes are restored exactly.
"""

from __future__ import annotations

import io
import json
import math
import types
import typing
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import numpy as np

from mvsr.core.errors import ContractValidationError, SchemaVersionError

SCHEMA_VERSION = 1
NPZ_META_KEY = "__meta__"
_NPZ_REF = "__npz__"

T = TypeVar("T")


def canonical_dumps(obj: Any) -> str:
    """Compact canonical JSON used for hashing (ids, checksums)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def pretty_dumps(obj: Any) -> str:
    """Human-readable JSON for files on disk (same content as `canonical_dumps`)."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


# --------------------------------------------------------------------------- encode


def encode(value: Any, sink: dict[str, np.ndarray[Any, Any]] | None = None, path: str = "") -> Any:
    """Contract value -> JSON-compatible Python objects.

    With `sink`, ndarrays are moved into `sink[path]` and replaced by a placeholder (NPZ mode).
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not math.isfinite(f):
            raise ContractValidationError(
                f"non-finite float at '{path or '<root>'}' cannot be JSON"
            )
        return f
    if isinstance(value, np.ndarray):
        if sink is not None:
            key = path or "array"
            if key in sink:
                raise ContractValidationError(f"duplicate NPZ key {key}")
            sink[key] = value
            return {_NPZ_REF: key}
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ContractValidationError(f"non-finite array at '{path}' cannot be JSON; use NPZ")
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: encode(getattr(value, f.name), sink, _join(path, f.name)) for f in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(k): encode(v, sink, _join(path, str(k))) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v, sink, _join(path, str(i))) for i, v in enumerate(value)]
    raise ContractValidationError(f"cannot encode {type(value).__name__} at '{path}'")


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _top(obj: Any, body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ContractValidationError("top-level object must be a dataclass")
    out = dict(body)
    if out.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise SchemaVersionError(f"{type(obj).__name__} has schema_version {out['schema_version']}")
    out["schema_version"] = SCHEMA_VERSION
    out["type"] = type(obj).__name__
    return out


def to_json_dict(obj: Any) -> dict[str, Any]:
    return _top(obj, encode(obj))


def to_json(obj: Any) -> str:
    return pretty_dumps(to_json_dict(obj))


def to_npz_bytes(obj: Any) -> bytes:
    sink: dict[str, np.ndarray[Any, Any]] = {}
    meta = _top(obj, encode(obj, sink))
    buf = io.BytesIO()
    payload: dict[str, Any] = dict(sink)
    if NPZ_META_KEY in payload:
        raise ContractValidationError(f"field path collides with {NPZ_META_KEY}")
    payload[NPZ_META_KEY] = np.frombuffer(canonical_dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(buf, **payload)
    return buf.getvalue()


# --------------------------------------------------------------------------- decode


def _check_top(cls: type[Any], data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SchemaVersionError(f"expected a JSON object for {cls.__name__}")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{cls.__name__}: unsupported schema_version {version!r} (expected {SCHEMA_VERSION})"
        )
    if data.get("type") != cls.__name__:
        raise SchemaVersionError(f"type tag {data.get('type')!r} != {cls.__name__}")
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names or k not in ("schema_version", "type")}


def from_json_dict(cls: type[T], data: Any) -> T:
    body = _check_top(cls, data)
    return typing.cast(T, decode(cls, body, None, ""))


def from_json(cls: type[T], text: str | bytes) -> T:
    return from_json_dict(cls, json.loads(text))


def from_npz_bytes(cls: type[T], blob: bytes) -> T:
    with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta_raw = arrays.pop(NPZ_META_KEY, None)
    if meta_raw is None:
        raise SchemaVersionError("NPZ has no __meta__ entry")
    meta = json.loads(meta_raw.tobytes().decode("utf-8"))
    body = _check_top(cls, meta)
    out = decode(cls, body, arrays, "")
    return typing.cast(T, out)


def _ndarray_dtype(tp: Any) -> np.dtype[Any] | None:
    args = get_args(tp)
    if len(args) == 2:
        dargs = get_args(args[1])
        if dargs and isinstance(dargs[0], type) and issubclass(dargs[0], np.generic):
            return np.dtype(dargs[0])
    return None


def decode(tp: Any, value: Any, arrays: dict[str, np.ndarray[Any, Any]] | None, path: str) -> Any:
    """JSON value -> instance of annotation `tp` (recursive)."""
    origin = get_origin(tp)
    args = get_args(tp)
    if tp is Any:
        return value
    if origin is Union or origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        options = [a for a in args if a is not type(None)]
        errors: list[str] = []
        for option in options:
            try:
                return decode(option, value, arrays, path)
            except (ContractValidationError, TypeError, ValueError, KeyError) as exc:
                errors.append(str(exc))
        raise ContractValidationError(f"'{path}': no union member matched ({'; '.join(errors)})")
    if origin is Literal:
        if value not in args:
            raise ContractValidationError(f"'{path}': {value!r} not in {args}")
        return value
    if tp is np.ndarray or origin is np.ndarray:
        dtype = _ndarray_dtype(tp)
        if isinstance(value, dict) and _NPZ_REF in value:
            if arrays is None or value[_NPZ_REF] not in arrays:
                raise SchemaVersionError(f"'{path}': missing NPZ entry {value.get(_NPZ_REF)}")
            arr = arrays[value[_NPZ_REF]]
            if dtype is not None and arr.dtype != dtype:
                raise ContractValidationError(f"'{path}': dtype {arr.dtype} != {dtype}")
            return arr
        return np.asarray(value, dtype=dtype)
    if isinstance(tp, type) and issubclass(tp, Enum):
        return tp(value)
    if isinstance(tp, type) and is_dataclass(tp):
        if not isinstance(value, dict):
            raise ContractValidationError(f"'{path}': expected object for {tp.__name__}")
        hints = get_type_hints(tp)
        known = {f.name for f in fields(tp)}
        unknown = set(value) - known
        if unknown:
            raise ContractValidationError(f"'{path}': unknown fields {sorted(unknown)}")
        kwargs = {
            f.name: decode(hints[f.name], value[f.name], arrays, _join(path, f.name))
            for f in fields(tp)
            if f.init and f.name in value
        }
        return tp(**kwargs)
    if origin in (dict, Mapping):
        if not isinstance(value, dict):
            raise ContractValidationError(f"'{path}': expected object")
        kt, vt = args if args else (Any, Any)
        return {
            _decode_key(kt, k, path): decode(vt, v, arrays, _join(path, k))
            for k, v in value.items()
        }
    if origin is tuple:
        if not isinstance(value, list):
            raise ContractValidationError(f"'{path}': expected array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                decode(args[0], v, arrays, _join(path, str(i))) for i, v in enumerate(value)
            )
        if len(args) != len(value):
            raise ContractValidationError(f"'{path}': expected {len(args)} items")
        return tuple(
            decode(a, v, arrays, _join(path, str(i)))
            for i, (a, v) in enumerate(zip(args, value, strict=True))
        )
    if origin is list:
        if not isinstance(value, list):
            raise ContractValidationError(f"'{path}': expected array")
        (item,) = args if args else (Any,)
        return [decode(item, v, arrays, _join(path, str(i))) for i, v in enumerate(value)]
    if tp is bool:
        if not isinstance(value, bool):
            raise ContractValidationError(f"'{path}': expected bool")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractValidationError(f"'{path}': expected int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractValidationError(f"'{path}': expected number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ContractValidationError(f"'{path}': expected str, got {value!r}")
        return value
    raise ContractValidationError(f"'{path}': unsupported annotation {tp!r}")


def _decode_key(kt: Any, key: str, path: str) -> Any:
    if kt is int:
        try:
            return int(key)
        except ValueError as exc:
            raise ContractValidationError(f"'{path}': key {key!r} is not an int") from exc
    return key
