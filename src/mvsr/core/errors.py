"""Error types shared by all modules (subplans/00_FOUNDATION.md F2)."""

from __future__ import annotations


class MvsrError(Exception):
    """Base class of all MVStreamRefine errors."""


class ContractValidationError(MvsrError, ValueError):
    """A value violates a rule of contracts/DATA_CONTRACTS.md (shape, dtype, range, invariant)."""


class FrameMismatchError(ContractValidationError):
    """Transform frames do not chain (`a.src != b.dst`) or do not match the expected frames."""


class InvalidPoseError(ContractValidationError):
    """A 4x4 matrix is not a valid rigid transform (non-finite, non-orthonormal, det != +1)."""


class SchemaVersionError(MvsrError):
    """A persisted object has a missing or unsupported `schema_version` (or wrong type tag)."""


class ChecksumMismatchError(MvsrError):
    """A persisted file's sha256 differs from the checksum recorded in its parent manifest."""


class DiskBudgetError(MvsrError):
    """Free disk space is below the configured reserve plus the estimated session size."""


class SessionFormatError(MvsrError):
    """A session directory violates contracts/SESSION_FORMAT.md (missing file, bad record)."""


class DeviceError(MvsrError):
    """A hardware device (RealSense, RM65) failed, disconnected or is misconfigured."""


class ClockModelError(MvsrError):
    """Recorded camera clocks cannot be mapped to HOST_MONOTONIC (reset, reorder, bad fit)."""
