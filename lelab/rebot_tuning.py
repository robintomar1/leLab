# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Position-loop gains for the reBot B601 follower's Damiao motors.

In position-velocity mode a Damiao motor closes the position error with its own
loop, whose gain is the KP_APR register. At the factory value (54) the follower
trails a moving target by about 10 / KP_APR seconds on the 4310 wrist motors and
40 / KP_APR on the 4340P motors (the ratio of their gearboxes): roughly 0.2 s
and 0.7 s, which is what makes teleoperation feel slow. The gains below bring
every arm joint to about 55 ms, measured on the arm with a 0.5 Hz sine:

    4310  (wrist_roll)                KP_APR 200 -> 57 ms
    4340P (shoulder_pan, elbow_flex)  KP_APR 800 -> 53-57 ms

The gains are written to the motors' RAM, never to flash: a power cycle brings
the factory values back. The gripper runs in force-position mode and is left
alone.
"""

import logging
import time

logger = logging.getLogger(__name__)

RID_KP_APR = 27
_WRITE_ATTEMPTS = 5
_READ_TIMEOUT_MS = 500

POSITION_KP: dict[str, float] = {
    "shoulder_pan": 800.0,
    "shoulder_lift": 800.0,
    "elbow_flex": 800.0,
    "wrist_flex": 200.0,
    "wrist_yaw": 200.0,
    "wrist_roll": 200.0,
}


def _open_bus(port: str, baud: int):
    from motorbridge import Controller

    return Controller.from_dm_serial(serial_port=port, baud=baud)


def _write_gain(motor, value: float) -> None:
    """Write KP_APR and check it reads back. The acknowledgement is occasionally missed on a busy bus."""
    last_error: Exception | None = None
    for _ in range(_WRITE_ATTEMPTS):
        try:
            motor.write_register_f32(RID_KP_APR, value)
            stored = motor.get_register_f32(RID_KP_APR, _READ_TIMEOUT_MS)
            if abs(stored - value) <= 1e-3 * max(1.0, abs(value)):
                return
            last_error = RuntimeError(f"KP_APR reads {stored:g} after writing {value:g}")
        except Exception as e:
            last_error = e
        time.sleep(0.05)
    raise RuntimeError(f"Could not set the position-loop gain: {last_error}")


def apply_position_gains(port: str, baud: int = 921600, gains: dict[str, float] | None = None) -> None:
    """Set the follower's position-loop gains, on a bus of its own.

    Call it before the follower is connected: the port must be free, and the
    motors are still disabled. Every way of driving the arm then sees the same
    response, including the `lerobot-rollout` subprocess, which builds its own
    robot and has no hook for this.
    """
    from lerobot.robots.rebot_b601_follower.motor_family import DM_PROFILE

    gains = POSITION_KP if gains is None else gains
    bus = _open_bus(port, baud)
    motors = []
    try:
        for joint, value in gains.items():
            send_id, recv_id = DM_PROFILE.motor_can_ids[joint]
            motor = bus.add_damiao_motor(send_id, recv_id, DM_PROFILE.motor_models[joint])
            motors.append(motor)
            try:
                _write_gain(motor, value)
            except RuntimeError as e:
                raise RuntimeError(f"{joint}: {e}") from e
        logger.info(f"reBot follower position-loop gains set: {gains}")
    finally:
        for motor in motors:
            motor.close()
        bus.close()
