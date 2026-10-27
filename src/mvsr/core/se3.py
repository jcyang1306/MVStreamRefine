"""SO(3)/SE(3) math and the frame-checked `PoseSE3` (contracts/COORDINATES_AND_TIMING.md).

Conventions (immutable, MASTER_PLAN.md §3):
- `T_A_B` maps homogeneous column points from B to A: `p_A = T_A_B @ p_B`.
- se(3) vector order `xi = [rho, phi] = [tx, ty, tz, rx, ry, rz]`; `se3_exp`/`se3_log` are the
  exact SE(3) maps (with the left Jacobian V), not decoupled.
- Left perturbation `T = Exp(delta) @ T_bar`; covariances are expressed in the `dst` frame.
- Quaternions on disk are `xyzw` with canonical `w >= 0`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mvsr.core.errors import ContractValidationError, FrameMismatchError, InvalidPoseError

F64 = npt.NDArray[np.float64]

#: Rotation orthonormality tolerance `|R^T R - I|_max` and `|det R - 1|` (DATA_CONTRACTS.md).
ORTHONORMAL_ATOL = 1e-6
#: Below this rotation angle (rad) closed-form coefficients are replaced by Taylor series;
#: the series are truncated after the theta^4 term (error < 1e-18 at the threshold).
SMALL_ANGLE_RAD = 1e-2
#: Within this distance of pi (rad) `so3_log` extracts the axis from the symmetric part of R.
NEAR_PI_RAD = 1e-2

_I3 = np.eye(3, dtype=np.float64)


def _vec(x: npt.ArrayLike, n: int, name: str) -> F64:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (n,):
        raise ContractValidationError(f"{name} must have shape ({n},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ContractValidationError(f"{name} contains non-finite values")
    return arr


def _mat(x: npt.ArrayLike, shape: tuple[int, int], name: str) -> F64:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != shape:
        raise ContractValidationError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ContractValidationError(f"{name} contains non-finite values")
    return arr


def _frozen(arr: F64) -> F64:
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


# --------------------------------------------------------------------------- so(3)


def hat3(w: npt.ArrayLike) -> F64:
    """Skew-symmetric matrix `[w]x` with `[w]x @ v == cross(w, v)`."""
    x, y, z = _vec(w, 3, "w")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def vee3(W: npt.ArrayLike) -> F64:
    """Inverse of `hat3` (uses the antisymmetric part)."""
    M = _mat(W, (3, 3), "W")
    return np.array(
        [0.5 * (M[2, 1] - M[1, 2]), 0.5 * (M[0, 2] - M[2, 0]), 0.5 * (M[1, 0] - M[0, 1])],
        dtype=np.float64,
    )


def so3_exp(phi: npt.ArrayLike) -> F64:
    """Rodrigues: rotation vector (rad) -> rotation matrix."""
    p = _vec(phi, 3, "phi")
    th2 = float(p @ p)
    th = math.sqrt(th2)
    W = hat3(p)
    if th < SMALL_ANGLE_RAD:
        a = 1.0 - th2 / 6.0 + th2 * th2 / 120.0
        b = 0.5 - th2 / 24.0 + th2 * th2 / 720.0
    else:
        a = math.sin(th) / th
        b = (1.0 - math.cos(th)) / th2
    out: F64 = _I3 + a * W + b * (W @ W)
    return out


def so3_log(R: npt.ArrayLike) -> F64:
    """Rotation matrix -> rotation vector with angle in [0, pi]; stable near 0 and pi."""
    M = _mat(R, (3, 3), "R")
    cos_t = 0.5 * (float(np.trace(M)) - 1.0)
    v = 2.0 * vee3(M)  # = 2 sin(theta) n
    sin_t = 0.5 * float(np.linalg.norm(v))
    th = math.atan2(sin_t, cos_t)
    if th < SMALL_ANGLE_RAD:
        th2 = th * th
        scale = 0.5 * (1.0 + th2 / 6.0 + 7.0 * th2 * th2 / 360.0)  # theta / (2 sin theta)
        out: F64 = scale * v
        return out
    if math.pi - th < NEAR_PI_RAD:
        # (R + R^T)/2 - cos(theta) I = (1 - cos(theta)) n n^T; take the best-conditioned column.
        B = 0.5 * (M + M.T) - cos_t * _I3
        k = int(np.argmax(np.diag(B)))
        n = B[:, k] / float(np.linalg.norm(B[:, k]))
        if float(n @ v) < 0.0:
            n = -n
        out = th * n
        return out
    out = (th / (2.0 * sin_t)) * v
    return out


def so3_left_jacobian(phi: npt.ArrayLike) -> F64:
    """J_l(phi) with `Exp(phi + d) ~= Exp(J_l(phi) d) @ Exp(phi)`."""
    p = _vec(phi, 3, "phi")
    th2 = float(p @ p)
    th = math.sqrt(th2)
    W = hat3(p)
    if th < SMALL_ANGLE_RAD:
        a = 0.5 - th2 / 24.0 + th2 * th2 / 720.0
        b = 1.0 / 6.0 - th2 / 120.0 + th2 * th2 / 5040.0
    else:
        a = (1.0 - math.cos(th)) / th2
        b = (th - math.sin(th)) / (th2 * th)
    out: F64 = _I3 + a * W + b * (W @ W)
    return out


def so3_left_jacobian_inv(phi: npt.ArrayLike) -> F64:
    """Inverse of `so3_left_jacobian` (valid for |phi| < 2 pi)."""
    p = _vec(phi, 3, "phi")
    th2 = float(p @ p)
    th = math.sqrt(th2)
    W = hat3(p)
    if th < SMALL_ANGLE_RAD:
        d = 1.0 / 12.0 + th2 / 720.0 + th2 * th2 / 30240.0
    else:
        half = 0.5 * th
        d = (1.0 - half * math.cos(half) / math.sin(half)) / th2
    out: F64 = _I3 - 0.5 * W + d * (W @ W)
    return out


# --------------------------------------------------------------------------- se(3)


def hat6(xi: npt.ArrayLike) -> F64:
    """`xi = [rho, phi]` -> 4x4 twist matrix `[[phi^, rho], [0, 0]]`."""
    x = _vec(xi, 6, "xi")
    X = np.zeros((4, 4), dtype=np.float64)
    X[:3, :3] = hat3(x[3:])
    X[:3, 3] = x[:3]
    return X


def vee6(X: npt.ArrayLike) -> F64:
    M = _mat(X, (4, 4), "X")
    return np.concatenate([M[:3, 3], vee3(M[:3, :3])])


def se3_exp(xi: npt.ArrayLike) -> F64:
    """Exact SE(3) exponential: `[[Exp(phi), J_l(phi) rho], [0, 1]]`."""
    x = _vec(xi, 6, "xi")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = so3_exp(x[3:])
    T[:3, 3] = so3_left_jacobian(x[3:]) @ x[:3]
    return T


def se3_log(T: npt.ArrayLike) -> F64:
    """Exact SE(3) logarithm -> `[rho, phi]` with |phi| <= pi."""
    M = _mat(T, (4, 4), "T")
    phi = so3_log(M[:3, :3])
    rho = so3_left_jacobian_inv(phi) @ M[:3, 3]
    return np.concatenate([rho, phi])


def se3_adjoint(T: npt.ArrayLike) -> F64:
    """Ad_T for `[rho, phi]` ordering: `T @ Exp(xi) @ T^-1 = Exp(Ad_T @ xi)`."""
    M = _mat(T, (4, 4), "T")
    R = M[:3, :3]
    Ad = np.zeros((6, 6), dtype=np.float64)
    Ad[:3, :3] = R
    Ad[:3, 3:] = hat3(M[:3, 3]) @ R
    Ad[3:, 3:] = R
    return Ad


def _q_coefficients(th: float) -> tuple[float, float, float]:
    th2 = th * th
    if th < SMALL_ANGLE_RAD:
        c1 = 1.0 / 6.0 - th2 / 120.0 + th2 * th2 / 5040.0
        c2 = 1.0 / 24.0 - th2 / 720.0 + th2 * th2 / 40320.0
        c3 = 1.0 / 120.0 - th2 / 2520.0 + th2 * th2 / 120960.0
        return c1, c2, c3
    s, c = math.sin(th), math.cos(th)
    c1 = (th - s) / (th2 * th)
    c2 = (th2 + 2.0 * c - 2.0) / (2.0 * th2 * th2)
    c3 = (2.0 * th - 3.0 * s + th * c) / (2.0 * th2 * th2 * th)
    return c1, c2, c3


def _se3_q(rho: F64, phi: F64) -> F64:
    P = hat3(rho)
    W = hat3(phi)
    c1, c2, c3 = _q_coefficients(float(np.linalg.norm(phi)))
    WP = W @ P
    PW = P @ W
    WPW = WP @ W
    WW = W @ W
    out: F64 = (
        0.5 * P
        + c1 * (WP + PW + WPW)
        + c2 * (WW @ P + PW @ W - 3.0 * WPW)
        + c3 * (WPW @ W + W @ WPW)
    )
    return out


def se3_left_jacobian(xi: npt.ArrayLike) -> F64:
    """6x6 J_l(xi) with `Exp(xi + d) ~= Exp(J_l(xi) d) @ Exp(xi)` (Barfoot's Q block)."""
    x = _vec(xi, 6, "xi")
    J = so3_left_jacobian(x[3:])
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = J
    out[:3, 3:] = _se3_q(x[:3], x[3:])
    out[3:, 3:] = J
    return out


def se3_left_jacobian_inv(xi: npt.ArrayLike) -> F64:
    x = _vec(xi, 6, "xi")
    Ji = so3_left_jacobian_inv(x[3:])
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = Ji
    out[:3, 3:] = -Ji @ _se3_q(x[:3], x[3:]) @ Ji
    out[3:, 3:] = Ji
    return out


# --------------------------------------------------------------------------- quaternions


def rotation_to_quat_xyzw(R: npt.ArrayLike) -> F64:
    """Rotation matrix -> unit quaternion `[x, y, z, w]`, canonical `w >= 0`.

    For `w == 0` (180 deg rotations) the first non-zero of x, y, z is made positive so the
    representation is unique.
    """
    M = _mat(R, (3, 3), "R")
    tr = float(np.trace(M))
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = np.array(
            [(M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s, 0.25 * s]
        )
    else:
        i = int(np.argmax(np.diag(M)))
        if i == 0:
            s = 2.0 * math.sqrt(max(1.0 + M[0, 0] - M[1, 1] - M[2, 2], 0.0))
            q = np.array(
                [
                    0.25 * s,
                    (M[0, 1] + M[1, 0]) / s,
                    (M[0, 2] + M[2, 0]) / s,
                    (M[2, 1] - M[1, 2]) / s,
                ]
            )
        elif i == 1:
            s = 2.0 * math.sqrt(max(1.0 + M[1, 1] - M[0, 0] - M[2, 2], 0.0))
            q = np.array(
                [
                    (M[0, 1] + M[1, 0]) / s,
                    0.25 * s,
                    (M[1, 2] + M[2, 1]) / s,
                    (M[0, 2] - M[2, 0]) / s,
                ]
            )
        else:
            s = 2.0 * math.sqrt(max(1.0 + M[2, 2] - M[0, 0] - M[1, 1], 0.0))
            q = np.array(
                [
                    (M[0, 2] + M[2, 0]) / s,
                    (M[1, 2] + M[2, 1]) / s,
                    0.25 * s,
                    (M[1, 0] - M[0, 1]) / s,
                ]
            )
    q = q.astype(np.float64) / float(np.linalg.norm(q))
    return canonical_quat_xyzw(q)


def canonical_quat_xyzw(q: npt.ArrayLike) -> F64:
    """Sign-canonicalize a unit quaternion (`w > 0`, or first non-zero of x, y, z > 0 if w == 0)."""
    v = _vec(q, 4, "q").copy()
    if v[3] < 0.0:
        return -v
    if v[3] == 0.0:
        nonzero = np.flatnonzero(v[:3])
        if nonzero.size and v[int(nonzero[0])] < 0.0:
            return -v
    return v


def quat_xyzw_to_rotation(q: npt.ArrayLike) -> F64:
    """Unit quaternion `[x, y, z, w]` -> rotation matrix. Norm must be 1 within 1e-6."""
    v = _vec(q, 4, "q")
    n = float(np.linalg.norm(v))
    if abs(n - 1.0) > 1e-6:
        raise ContractValidationError(f"quaternion norm {n} is not 1 (tolerance 1e-6)")
    x, y, z, w = v / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- validation


def validate_rotation(R: npt.ArrayLike, atol: float = ORTHONORMAL_ATOL) -> F64:
    try:
        M = _mat(R, (3, 3), "R")
    except ContractValidationError as exc:
        raise InvalidPoseError(str(exc)) from exc
    err = float(np.max(np.abs(M.T @ M - _I3)))
    if err > atol:
        raise InvalidPoseError(f"rotation not orthonormal: max|R^T R - I| = {err:.3g} > {atol}")
    det = float(np.linalg.det(M))
    if abs(det - 1.0) > atol:
        raise InvalidPoseError(f"rotation det = {det!r} is not +1 (atol {atol})")
    return M


def validate_rigid(T: npt.ArrayLike, atol: float = ORTHONORMAL_ATOL) -> F64:
    """Return `T` as float64 4x4 after checking finiteness, rotation and last row `[0,0,0,1]`."""
    try:
        M = _mat(T, (4, 4), "T")
    except ContractValidationError as exc:
        raise InvalidPoseError(str(exc)) from exc
    if not np.array_equal(M[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise InvalidPoseError(f"last row must be [0, 0, 0, 1], got {M[3].tolist()}")
    validate_rotation(M[:3, :3], atol)
    return M


def _validate_covariance(C: npt.ArrayLike) -> F64:
    M = _mat(C, (6, 6), "covariance")
    scale = max(1.0, float(np.max(np.abs(M))))
    if float(np.max(np.abs(M - M.T))) > 1e-9 * scale:
        raise ContractValidationError("covariance is not symmetric")
    min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (M + M.T))))
    if min_eig < -1e-9 * scale:
        raise ContractValidationError(f"covariance is not PSD (min eigenvalue {min_eig:.3g})")
    return M


def _rigid_inverse(M: F64) -> F64:
    R = M[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ M[:3, 3]
    return out


# --------------------------------------------------------------------------- PoseSE3


@dataclass(frozen=True, eq=False)
class PoseSE3:
    """Rigid transform `T_dst_src` (maps points in `src` to `dst`) with optional covariance.

    `covariance` is 6x6 over `[tx, ty, tz, rx, ry, rz]` for the left perturbation
    `T = Exp(delta) @ T_bar` (delta expressed in `dst`). `None` means "not modelled"; it is
    never treated as zero. Composition returns a covariance only when both operands have one
    (assumed independent); inversion maps it with the adjoint.
    """

    matrix: F64
    dst: str
    src: str
    covariance: F64 | None = None

    def __post_init__(self) -> None:
        for name in ("dst", "src"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractValidationError(f"PoseSE3.{name} must be a non-empty str")
        object.__setattr__(self, "matrix", _frozen(validate_rigid(self.matrix)))
        if self.covariance is not None:
            object.__setattr__(self, "covariance", _frozen(_validate_covariance(self.covariance)))

    # constructors ---------------------------------------------------------
    @classmethod
    def identity(cls, dst: str, src: str) -> PoseSE3:
        return cls(np.eye(4), dst, src)

    @classmethod
    def from_rt(
        cls,
        R: npt.ArrayLike,
        t: npt.ArrayLike,
        dst: str,
        src: str,
        covariance: npt.ArrayLike | None = None,
    ) -> PoseSE3:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _mat(R, (3, 3), "R")
        T[:3, 3] = _vec(t, 3, "t")
        cov = None if covariance is None else np.asarray(covariance, dtype=np.float64)
        return cls(T, dst, src, cov)

    @classmethod
    def from_quat_xyzw(cls, q: npt.ArrayLike, t: npt.ArrayLike, dst: str, src: str) -> PoseSE3:
        return cls.from_rt(quat_xyzw_to_rotation(q), t, dst, src)

    # accessors ------------------------------------------------------------
    @property
    def R(self) -> F64:
        return self.matrix[:3, :3]

    @property
    def t(self) -> F64:
        return self.matrix[:3, 3]

    @property
    def name(self) -> str:
        return f"T_{self.dst}_{self.src}"

    def quat_xyzw(self) -> F64:
        return rotation_to_quat_xyzw(self.R)

    def log(self) -> F64:
        return se3_log(self.matrix)

    def with_covariance(self, covariance: npt.ArrayLike | None) -> PoseSE3:
        cov = None if covariance is None else np.asarray(covariance, dtype=np.float64)
        return PoseSE3(self.matrix, self.dst, self.src, cov)

    # algebra --------------------------------------------------------------
    def inverse(self) -> PoseSE3:
        inv = _rigid_inverse(self.matrix)
        cov = None
        if self.covariance is not None:
            Ad = se3_adjoint(inv)
            cov = Ad @ self.covariance @ Ad.T
            cov = 0.5 * (cov + cov.T)
        return PoseSE3(inv, self.src, self.dst, cov)

    def __matmul__(self, other: PoseSE3) -> PoseSE3:
        if not isinstance(other, PoseSE3):
            return NotImplemented
        if self.src != other.dst:
            raise FrameMismatchError(
                f"cannot compose {self.name} @ {other.name}: '{self.src}' != '{other.dst}'"
            )
        M = self.matrix @ other.matrix
        cov = None
        if self.covariance is not None and other.covariance is not None:
            Ad = se3_adjoint(self.matrix)
            cov = self.covariance + Ad @ other.covariance @ Ad.T
            cov = 0.5 * (cov + cov.T)
        return PoseSE3(M, self.dst, other.src, cov)

    def transform_points(self, points: npt.ArrayLike) -> F64:
        """`(N, 3)` points in `src` -> float64 `(N, 3)` points in `dst`."""
        P = np.asarray(points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ContractValidationError(f"points must have shape (N, 3), got {P.shape}")
        out: F64 = P @ self.R.T + self.t
        return out

    def expect_frames(self, dst: str, src: str) -> PoseSE3:
        """Return self after asserting the frames, else raise `FrameMismatchError`."""
        if self.dst != dst or self.src != src:
            raise FrameMismatchError(f"expected T_{dst}_{src}, got {self.name}")
        return self

    # comparison -----------------------------------------------------------
    def allclose(self, other: PoseSE3, atol: float = 1e-9) -> bool:
        return (
            self.dst == other.dst
            and self.src == other.src
            and bool(np.allclose(self.matrix, other.matrix, atol=atol, rtol=0.0))
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PoseSE3):
            return NotImplemented
        if self.dst != other.dst or self.src != other.src:
            return False
        if not np.array_equal(self.matrix, other.matrix):
            return False
        if self.covariance is None or other.covariance is None:
            return self.covariance is None and other.covariance is None
        return bool(np.array_equal(self.covariance, other.covariance))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        t = ", ".join(f"{v:.6g}" for v in self.t)
        rotvec = so3_log(self.R)
        deg = math.degrees(float(np.linalg.norm(rotvec)))
        cov = "" if self.covariance is None else ", cov"
        return f"PoseSE3({self.name}, t=[{t}] m, rot={deg:.4g} deg{cov})"


# --------------------------------------------------------------------------- interpolation


def _check_interp(a: PoseSE3, b: PoseSE3, alpha: float) -> None:
    if a.dst != b.dst or a.src != b.src:
        raise FrameMismatchError(f"cannot interpolate {a.name} with {b.name}")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ContractValidationError(f"alpha must be in [0, 1] (no extrapolation), got {alpha}")


def interpolate_decoupled(a: PoseSE3, b: PoseSE3, alpha: float) -> PoseSE3:
    """Rotation slerp + translation lerp (robot TCP samples; COORDINATES_AND_TIMING.md).

    `alpha = 0` returns `a`, `alpha = 1` returns `b`; covariance is not propagated (None).
    """
    _check_interp(a, b, alpha)
    R = a.R @ so3_exp(alpha * so3_log(a.R.T @ b.R))
    t = (1.0 - alpha) * a.t + alpha * b.t
    return PoseSE3.from_rt(R, t, a.dst, a.src)


def interpolate_geodesic(a: PoseSE3, b: PoseSE3, alpha: float) -> PoseSE3:
    """SE(3) geodesic `a @ Exp(alpha * Log(a^-1 @ b))` (pose-graph use); covariance None."""
    _check_interp(a, b, alpha)
    delta = se3_log(_rigid_inverse(a.matrix) @ b.matrix)
    M = a.matrix @ se3_exp(alpha * delta)
    return PoseSE3(M, a.dst, a.src)
