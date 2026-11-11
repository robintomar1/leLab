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
"""Tests for the reBot B601 follower that feeds the target's velocity forward in MIT mode."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from lelab.rebot_ff_follower import (
    FOLLOWER_TYPE,
    RebotB601FeedforwardFollower,
    RebotB601FeedforwardFollowerConfig,
    TargetVelocityEstimator,
)
from lelab.robot_models import REBOT_MIT, get_robot_model, make_follower

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]


def test_estimator_converges_on_the_slope_of_a_steady_ramp() -> None:
    estimator = TargetVelocityEstimator(filter_s=0.04, max_velocity=150.0)
    velocity = 0.0
    for i in range(200):  # 1 s at 200 Hz, moving at 30 deg/s
        velocity = estimator.update(i * 0.005, 30.0 * i * 0.005)
    assert velocity == pytest.approx(30.0, abs=0.1)


def test_estimator_smooths_a_quantized_target() -> None:
    """A leader's reading that flips by one step is a huge instantaneous rate; the estimate stays small."""
    estimator = TargetVelocityEstimator(filter_s=0.04, max_velocity=150.0)
    worst = 0.0
    for i in range(400):
        worst = max(worst, abs(estimator.update(i * 0.005, 0.088 * (i % 2))))  # raw rate: ±17.6 deg/s
    assert worst < 3.0


def test_estimator_starts_from_rest_after_a_pause_and_never_exceeds_its_cap() -> None:
    estimator = TargetVelocityEstimator(filter_s=0.04, max_velocity=150.0)
    assert estimator.update(0.0, 0.0) == 0.0  # nothing to difference yet
    for i in range(1, 100):
        estimator.update(i * 0.005, 1000.0 * i * 0.005)  # 1000 deg/s
    assert estimator.velocity == 150.0
    # The next target arrives after a gap: a new motion, not a slow one.
    assert estimator.update(5.0, 0.0) == 0.0


class _Motor:
    def __init__(self, bus: _Bus, name: str) -> None:
        self.bus, self.name, self.position_rad = bus, name, 0.0

    def request_feedback(self) -> None:
        pass

    def get_state(self):
        return SimpleNamespace(pos=self.position_rad, vel=0.0, torq=0.0)

    def send_mit(self, pos, vel, kp, kd, tau) -> None:
        self.bus.sent.append((self.name, "mit", pos, vel, kp, kd, tau))

    def send_force_pos(self, pos, vlim, ratio) -> None:
        self.bus.sent.append((self.name, "force_pos", pos, vlim, ratio))


class _Bus:
    def __init__(self) -> None:
        self.sent: list[tuple] = []

    def poll_feedback_once(self) -> None:
        pass


@pytest.fixture
def follower(monkeypatch: pytest.MonkeyPatch):
    """The follower LeLab builds for rebot_mit, on fake motors, with a clock the test advances."""
    robot = make_follower(get_robot_model(REBOT_MIT), "/dev/ttyACM0", "arm")
    assert type(robot) is RebotB601FeedforwardFollower
    bus = _Bus()
    robot.bus = bus
    robot.motors = {name: _Motor(bus, name) for name in robot.motor_names}

    import lelab.rebot_ff_follower as module

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock.now)

    def send(positions: dict[str, float], dt: float = 0.005) -> dict[str, tuple]:
        clock.now += dt
        bus.sent.clear()
        robot.send_action({f"{name}.pos": value for name, value in positions.items()})
        return {entry[0]: entry[1:] for entry in bus.sent}

    return robot, send


def test_built_follower_is_the_same_arm_to_lerobot(follower) -> None:
    robot, _ = follower
    assert robot.name == "rebot_b601_follower"  # same calibration directory and dataset robot_type
    assert robot.config.type == FOLLOWER_TYPE
    assert robot.config.control_mode.value == "mit"
    assert robot.config.max_relative_target == 20.0


def test_mit_commands_carry_the_targets_velocity(follower) -> None:
    robot, send = follower
    rest = dict.fromkeys([*ARM_JOINTS, "gripper"], 0.0)
    sent = {}
    for i in range(200):  # shoulder_pan moves at +30 deg/s, the motors keep up
        position = 30.0 * i * 0.005
        robot.motors["shoulder_pan"].position_rad = math.radians(position)
        sent = send({**rest, "shoulder_pan": position})

    mode, position_rad, velocity_rad_s, kp, kd, torque = sent["shoulder_pan"]
    assert mode == "mit"
    assert position_rad == pytest.approx(math.radians(30.0 * 199 * 0.005))
    assert velocity_rad_s == pytest.approx(math.radians(30.0), rel=0.01)
    assert (kp, kd, torque) == (45.0, 12.0, 0.0)  # LeRobot's gains: the loop is no stiffer
    # A joint whose target is not moving is sent no velocity.
    assert sent["elbow_flex"][2] == 0.0
    # The gripper keeps LeRobot's force-position command.
    assert sent["gripper"][0] == "force_pos"


def test_feedforward_follows_the_leader_not_the_clipped_target(follower) -> None:
    """With the follower far from a still leader, the target is clipped to within
    max_relative_target of the arm. That clipped target moves as the arm moves: feeding
    its velocity forward would cancel the damping that slows the approach."""
    robot, send = follower
    rest = dict.fromkeys([*ARM_JOINTS, "gripper"], 0.0)
    sent = {}
    for i in range(100):
        # The leader holds -90 deg; the arm is approaching it at 60 deg/s from 0.
        arm = -60.0 * i * 0.005
        robot.motors["shoulder_lift"].position_rad = math.radians(arm)
        sent = send({**rest, "shoulder_lift": -90.0})

    _, position_rad, velocity_rad_s, *_ = sent["shoulder_lift"]
    assert math.degrees(position_rad) == pytest.approx(-60.0 * 99 * 0.005 - 20.0)  # arm - cap
    assert velocity_rad_s == pytest.approx(0.0, abs=1e-6)  # the leader is still


def test_no_feedforward_past_a_joint_limit(follower) -> None:
    robot, send = follower
    rest = dict.fromkeys([*ARM_JOINTS, "gripper"], 0.0)
    robot.motors["wrist_flex"].position_rad = math.radians(90.0)
    sent = {}
    for i in range(100):  # the leader keeps going past wrist_flex's +90 deg limit
        sent = send({**rest, "wrist_flex": 95.0 + 40.0 * i * 0.005})
    _, position_rad, velocity_rad_s, *_ = sent["wrist_flex"]
    assert math.degrees(position_rad) == pytest.approx(90.0)
    assert velocity_rad_s == pytest.approx(0.0, abs=1e-3)


def test_zero_feedforward_gain_is_lerobots_driver(follower) -> None:
    robot, send = follower
    robot.config.velocity_feedforward = 0.0
    rest = dict.fromkeys([*ARM_JOINTS, "gripper"], 0.0)
    sent = {}
    for i in range(50):
        robot.motors["wrist_roll"].position_rad = math.radians(30.0 * i * 0.005)
        sent = send({**rest, "wrist_roll": 30.0 * i * 0.005})
    assert sent["wrist_roll"][2] == 0.0


def test_rollout_runs_through_lelabs_entry_point_and_builds_the_same_follower(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inference has to drive the arm the way recording did: the subprocess is started
    through the module that registers LeLab's follower, and LeRobot's parser accepts the flags."""
    import lelab.rebot_tuning as tuning
    import lelab.rollout_entry as entry
    from lelab import rollout
    from lelab.rollout import InferenceRequest
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts import lerobot_rollout

    assert entry.main is lerobot_rollout.main

    checkpoint = tmp_path / "checkpoints" / "000050" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    ACTConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )._save_pretrained(checkpoint)

    commands: list[list[str]] = []

    def capture(cmd, **kwargs):
        commands.append(cmd)
        raise OSError("stop before spawning")

    gains: list = []
    monkeypatch.setattr(tuning, "apply_position_gains", lambda port, gains_, baud=921600: gains.append(port))
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(
        rollout, "setup_follower_calibration_file", lambda name, robot_model=None: "follower1"
    )
    monkeypatch.setattr(rollout.subprocess, "Popen", capture)
    rollout.handle_start_inference(
        InferenceRequest(
            follower_port="/dev/ttyACM0",
            follower_config="follower1.json",
            robot_model=REBOT_MIT,
            policy_ref=str(checkpoint),
        )
    )
    (command,) = commands
    assert command[:3] == [sys.executable, "-m", "lelab.rollout_entry"]
    assert gains == []  # MIT mode does not use the motors' position loop

    parsed = []

    class _StopBeforeHardwareError(Exception):
        pass

    def stop_before_hardware(cfg, _shutdown_event):
        parsed.append(cfg)
        raise _StopBeforeHardwareError

    monkeypatch.setattr(lerobot_rollout, "build_rollout_context", stop_before_hardware)
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", *command[3:]])
    with pytest.raises(_StopBeforeHardwareError):
        entry.main()

    (cfg,) = parsed
    assert type(cfg.robot) is RebotB601FeedforwardFollowerConfig
    assert cfg.robot.control_mode.value == "mit"
    assert cfg.robot.max_relative_target == 20.0
    assert cfg.robot.velocity_feedforward == 1.0
