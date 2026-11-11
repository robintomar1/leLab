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

"""The reBot B601 follower in MIT mode, with the target's velocity fed forward.

In MIT mode a Damiao motor produces

    torque = kp * (target_position - position) + kd * (target_velocity - velocity)

LeRobot's driver always sends target_velocity = 0, so the damping term resists
every motion: a joint following a moving target settles kd / kp seconds behind
it (about 0.13 s on the wrist, 0.27 s on the shoulder and elbow with LeRobot's
gains). Sending the velocity the target is actually moving at removes that lag
without raising either gain, so the loop is no stiffer than before.

The driver only receives positions, so the velocity is estimated here from
successive targets. This module defines its own config type: LeRobot builds the
device from the config's module, and `lelab.rollout_entry` imports it so the
rollout subprocess can build the same follower from `--robot.type`.
"""

import math
import time
from dataclasses import dataclass

from lerobot.lerobot_types import RobotAction
from lerobot.robots.config import RobotConfig
from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerRobotConfig
from lerobot.robots.rebot_b601_follower.motor_family import GRIPPER_MOTOR, ArmControlMode
from lerobot.robots.rebot_b601_follower.rebot_b601_follower import RebotB601Follower

FOLLOWER_TYPE = "lelab_rebot_b601_ff_follower"

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
    # Fraction of the target's velocity sent to the motors. 0 behaves like LeRobot's driver.
    velocity_feedforward: float = 1.0
    # Time constant of the velocity estimate's low-pass filter, in seconds.
    velocity_filter_s: float = 0.04
    # The feed-forward never asks for more than this, in degrees per second.
    max_feedforward_velocity: float = 150.0


class RebotB601FeedforwardFollower(RebotB601Follower):
    """reBot B601 follower whose MIT commands carry the target's velocity."""

    config_class = RebotB601FeedforwardFollowerConfig
    # Same arm: same calibration files, same robot_type on datasets.
    name = RebotB601Follower.name

    def __init__(self, config: RebotB601FeedforwardFollowerConfig):
        super().__init__(config)
        self._estimators = {
            motor: TargetVelocityEstimator(config.velocity_filter_s, config.max_feedforward_velocity)
            for motor in self.motor_names
        }
        self._target_velocity: dict[str, float] = dict.fromkeys(self.motor_names, 0.0)

    def send_action(self, action: RobotAction) -> RobotAction:
        # Estimated from the targets as asked for, before `max_relative_target` pulls
        # them towards the arm: a target clipped to "present position + cap" moves
        # with the arm, and feeding that velocity forward would cancel the damping
        # that is supposed to slow the approach.
        now = time.perf_counter()
        for key, value in action.items():
            motor = key.removesuffix(".pos")
            if key.endswith(".pos") and motor in self._estimators:
                low, high = self._joint_limits.get(motor, (-math.inf, math.inf))
                self._target_velocity[motor] = self._estimators[motor].update(now, max(low, min(high, value)))
        return super().send_action(action)

    def _send_joint(self, motor_name: str, position_deg: float, state=None) -> float:
        feedforward = self.config.velocity_feedforward
        if (
            motor_name == GRIPPER_MOTOR
            or self.config.control_mode is not ArmControlMode.MIT
            or not feedforward
        ):
            return super()._send_joint(motor_name, position_deg, state)

        limits = self._joint_limits.get(motor_name)
        if limits is not None:
            position_deg = max(limits[0], min(limits[1], position_deg))
        velocity_deg_s = feedforward * self._target_velocity[motor_name]
        self.motors[motor_name].send_mit(
            math.radians(self._public_to_motor_position(motor_name, position_deg)),
            math.radians(self._public_to_motor_position(motor_name, velocity_deg_s)),
            self._mit_kp[motor_name],
            self._mit_kd[motor_name],
            0.0,
        )
        return position_deg
