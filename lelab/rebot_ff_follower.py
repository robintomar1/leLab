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

"""The reBot B601 follower in MIT mode, with the feed-forward terms LeRobot's driver leaves at zero.

In MIT mode a Damiao motor produces

    torque = kp * (target_position - position) + kd * (target_velocity - velocity) + feedforward_torque

LeRobot's driver sends target_velocity = 0 and feedforward_torque = 0. Two things
follow, and this follower addresses each without raising either gain, so the loop
is no stiffer than before:

- Nothing carries the arm's weight: a loaded joint settles where the spring
  balances gravity, several degrees from its target. `gravity_compensation` sends
  the torque gravity asks of each joint (lelab/rebot_gravity.py) as the
  feed-forward torque.
- The damping term resists every motion: a joint following a moving target
  settles kd / kp seconds behind it (about 0.13 s on the wrist, 0.27 s on the
  shoulder and elbow with LeRobot's gains). `velocity_feedforward` sends the
  velocity the target is actually moving at, estimated from successive targets
  since the driver only receives positions.

This module defines its own config type: LeRobot builds the device from the
config's module, and `lelab.rollout_entry` imports it so the rollout subprocess
can build the same follower from `--robot.type`.
"""

import logging
import math
import time
from dataclasses import dataclass, field

from lerobot.lerobot_types import RobotAction
from lerobot.robots.config import RobotConfig
from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerRobotConfig
from lerobot.robots.rebot_b601_follower.motor_family import GRIPPER_MOTOR, ArmControlMode, MotorFamily
from lerobot.robots.rebot_b601_follower.rebot_b601_follower import RebotB601Follower
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.decorators import check_if_not_connected

from .rebot_gravity import JOINTS as GRAVITY_JOINTS, gravity_torques
from .utils.devices import MotorFaultError

FOLLOWER_TYPE = "lelab_rebot_b601_ff_follower"

# What a Damiao motor reports in its feedback once it has protected itself. It
# stops producing torque until the error is cleared, so the arm goes limp.
MOTOR_FAULTS = {
    0x8: "over-voltage",
    0x9: "under-voltage",
    0xA: "over-current",
    0xB: "driver over-temperature",
    0xC: "winding over-temperature",
    0xD: "communication lost",
    0xE: "overload",
}


_FEEDBACK_ATTEMPTS = 3


class _OncePerSecond(logging.Filter):
    """Lets one matching record through per second.

    LeRobot warns, with a formatted table, every time `max_relative_target` clips a
    target. With the bound doing its job as a speed limit that is every control
    cycle of a fast move: a hundred multi-line warnings a second, written from the
    control thread.
    """

    def __init__(self, needle: str) -> None:
        super().__init__()
        self._needle = needle
        self._last = float("-inf")

    def filter(self, record: logging.LogRecord) -> bool:
        if self._needle not in str(record.msg):
            return True
        now = time.monotonic()
        if now - self._last < 1.0:
            return False
        self._last = now
        return True


logging.getLogger("lerobot.robots.utils").addFilter(_OncePerSecond("Relative goal position magnitude"))


# Targets further apart in time than this are separate motions, not one to differentiate.
_MAX_TARGET_GAP_S = 0.25


class TargetVelocityEstimator:
    """Low-pass filtered rate of change of a stream of position targets.

    A leader arm's readings are quantized, so differencing two of them a few
    milliseconds apart is mostly noise; the filter trades a little delay in the
    feed-forward for a usable signal.
    """

    def __init__(self, filter_s: float, max_velocity: float) -> None:
        self._filter_s = filter_s
        self._max_velocity = max_velocity
        self._last_t: float | None = None
        self._last_position = 0.0
        self.velocity = 0.0

    def update(self, t: float, position: float) -> float:
        if self._last_t is None or not 0.0 < t - self._last_t <= _MAX_TARGET_GAP_S:
            self.velocity = 0.0
        else:
            dt = t - self._last_t
            raw = (position - self._last_position) / dt
            self.velocity += dt / (self._filter_s + dt) * (raw - self.velocity)
            self.velocity = max(-self._max_velocity, min(self._max_velocity, self.velocity))
        self._last_t, self._last_position = t, position
        return self.velocity


@RobotConfig.register_subclass(FOLLOWER_TYPE)
@dataclass
class RebotB601FeedforwardFollowerConfig(RebotB601FollowerRobotConfig):
    # Fraction of the model's gravity torque sent to the motors. 0 sends none.
    gravity_compensation: float = 1.0
    # Per-joint correction of the mass model, multiplied into the gravity torque.
    # The wrist_flex value was tuned on the arm with reBotArm_control_py's gravity
    # compensation demo (`tau_scale` in its rebotarm_dm.yaml).
    gravity_scale: dict[str, float] = field(default_factory=lambda: {"wrist_flex": 1.10})
    # The gravity torque fades in over this long after the first command, so a
    # resting arm is not jolted when a session starts.
    gravity_ramp_s: float = 1.0
    # Fraction of the target's velocity sent to the motors. 0 sends none.
    velocity_feedforward: float = 1.0
    # Time constant of the velocity estimate's low-pass filter, in seconds.
    velocity_filter_s: float = 0.04
    # The velocity feed-forward never asks for more than this, in degrees per second.
    max_feedforward_velocity: float = 150.0


class RebotB601FeedforwardFollower(RebotB601Follower):
    """reBot B601 follower whose MIT commands carry gravity torque and the target's velocity."""

    config_class = RebotB601FeedforwardFollowerConfig
    # Same arm: same calibration files, same robot_type on datasets.
    name = RebotB601Follower.name

    def __init__(self, config: RebotB601FeedforwardFollowerConfig):
        super().__init__(config)
        if config.gravity_compensation and config.motor_family is not MotorFamily.DM:
            raise ValueError("gravity_compensation uses the Damiao build's mass model")
        self._estimators = {
            motor: TargetVelocityEstimator(config.velocity_filter_s, config.max_feedforward_velocity)
            for motor in self.motor_names
        }
        self._target_velocity: dict[str, float] = dict.fromkeys(self.motor_names, 0.0)
        self._gravity_torque: dict[str, float] = {}
        self._first_command_at: float | None = None

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        self._first_command_at = None

    @property
    def _feedforward_active(self) -> bool:
        return self.config.control_mode is ArmControlMode.MIT

    def _gravity_feedforward(self, now: float, present_deg: dict[str, float]) -> dict[str, float]:
        """Torque per arm joint, in the motors' frame, that holds the arm where it is."""
        if self._first_command_at is None:
            self._first_command_at = now
        ramp = self.config.gravity_ramp_s
        fade = 1.0 if ramp <= 0 else min(1.0, (now - self._first_command_at) / ramp)
        # The mass model is written in the motors' own angles.
        angles = [math.radians(self._public_to_motor_position(j, present_deg[j])) for j in GRAVITY_JOINTS]
        scale = self.config.gravity_compensation * fade
        return {
            joint: scale * self.config.gravity_scale.get(joint, 1.0) * torque
            for joint, torque in zip(GRAVITY_JOINTS, gravity_torques(angles), strict=True)
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._feedforward_active:
            return super().send_action(action)

        now = time.perf_counter()
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # The velocity comes from the targets as asked for, before `max_relative_target`
        # pulls them towards the arm: a target clipped to "present position + cap" moves
        # with the arm, and feeding that velocity forward would cancel the damping that
        # is supposed to slow the approach.
        for motor, value in goal_pos.items():
            if motor in self._estimators:
                low, high = self._joint_limits.get(motor, (-math.inf, math.inf))
                self._target_velocity[motor] = self._estimators[motor].update(now, max(low, min(high, value)))

        # One read of the arm serves the bound on the target, the gravity torque, and
        # noticing a motor that has shut itself down.
        states = None
        self._gravity_torque = {}
        if self.config.max_relative_target is not None or self.config.gravity_compensation:
            states = self._read_feedback()
            present_pos = self._present_pos(states)
            if self.config.gravity_compensation:
                self._gravity_torque = self._gravity_feedforward(now, present_pos)
            if self.config.max_relative_target is not None:
                goal_present_pos = {key: (goal, present_pos.get(key, goal)) for key, goal in goal_pos.items()}
                relative_limits = self.config.max_relative_target
                if isinstance(relative_limits, dict):
                    relative_limits = {key: relative_limits[key] for key in goal_present_pos}
                goal_pos = ensure_safe_goal_position(goal_present_pos, relative_limits)

        sent_pos = {}
        for motor_name, position_deg in goal_pos.items():
            if motor_name in self.motors:
                state = states.get(motor_name) if states is not None else None
                sent_pos[motor_name] = self._send_joint(motor_name, position_deg, state)
        return {f"{motor}.pos": val for motor, val in sent_pos.items()}

    def _read_feedback(self) -> dict:
        """The motors' states, for every caller: observations as well as commands.

        One reply lost on a busy bus is retried, not fatal: an error here ends the
        session, and ending it drops the arm. A motor that reports it has shut
        itself down does end it, in any control mode, with the reason.
        """
        for attempt in range(_FEEDBACK_ATTEMPTS):
            try:
                states = super()._read_feedback()
                break
            except RuntimeError:
                if attempt == _FEEDBACK_ATTEMPTS - 1:
                    raise
        self._raise_on_fault(states)
        return states

    def _raise_on_fault(self, states: dict) -> None:
        faults = {
            name: MOTOR_FAULTS[code]
            for name, state in states.items()
            if (code := getattr(state, "status_code", None)) in MOTOR_FAULTS
        }
        if not faults:
            return
        temperatures = ", ".join(
            f"{name} {getattr(state, 't_mos', float('nan')):.0f}/{getattr(state, 't_rotor', float('nan')):.0f}"
            for name, state in states.items()
        )
        raise MotorFaultError(
            "The follower's motors protected themselves and the arm is limp: "
            + ", ".join(f"{name} reports {fault}" for name, fault in faults.items())
            + f". Driver/winding temperatures in °C: {temperatures}. "
            "Power-cycle the arm, or start a new session, to clear it."
        )

    def _send_joint(self, motor_name: str, position_deg: float, state=None) -> float:
        if motor_name == GRIPPER_MOTOR or not self._feedforward_active:
            return super()._send_joint(motor_name, position_deg, state)

        limits = self._joint_limits.get(motor_name)
        if limits is not None:
            position_deg = max(limits[0], min(limits[1], position_deg))
        velocity_deg_s = self.config.velocity_feedforward * self._target_velocity[motor_name]
        self.motors[motor_name].send_mit(
            math.radians(self._public_to_motor_position(motor_name, position_deg)),
            math.radians(self._public_to_motor_position(motor_name, velocity_deg_s)),
            self._mit_kp[motor_name],
            self._mit_kd[motor_name],
            self._gravity_torque.get(motor_name, 0.0),
        )
        return position_deg
