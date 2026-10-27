"""Robotic_Arm.rm_robot_interface test double (control flow / error paths only, ADR-0002)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


class FakeArm:
    def __init__(self, sdk: FakeRM) -> None:
        self.sdk = sdk
        self.calls = 0
        self.deleted = False
        self.lock = threading.Lock()

    def rm_create_robot_arm(self, ip: str, port: int) -> SimpleNamespace:
        return SimpleNamespace(id=1 if self.sdk.connect_ok else -1)

    def rm_get_current_arm_state(self) -> tuple[int, dict[str, Any]]:
        with self.lock:
            n = self.calls
            self.calls += 1
        if self.sdk.fail(n):
            return -2, {}
        return 0, {
            "joint": [10.0, -20.0, 30.0, 0.0, 45.0, 90.0],
            "pose": [0.3, -0.1, 0.4 + 1e-5 * n, 3.1, 0.05, -1.2],
            "err": {},
        }

    def rm_get_robot_info(self) -> tuple[int, dict[str, Any]]:
        return 0, {"arm_dof": 6, "arm_model": "RM_65"}

    def rm_get_arm_software_info(self) -> tuple[int, dict[str, Any]]:
        return 0, {"product_version": "fake"}

    def rm_delete_robot_arm(self) -> int:
        self.deleted = True
        return 0


class FakeRM:
    def __init__(
        self, *, connect_ok: bool = True, fail: Callable[[int], bool] = lambda n: False
    ) -> None:
        self.connect_ok = connect_ok
        self.fail = fail
        self.arms: list[FakeArm] = []
        self.rm_thread_mode_e = SimpleNamespace(RM_TRIPLE_MODE_E=3)

    def RoboticArm(self, mode: Any) -> FakeArm:
        arm = FakeArm(self)
        self.arms.append(arm)
        return arm
