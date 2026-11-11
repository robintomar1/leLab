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
from lelab.robot_models import REBOT_MIT, REBOT_MIT_FF, get_robot_model, make_follower

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
    """The follower LeLab builds for rebot_mit_ff, on fake motors, with a clock the test advances."""
    robot = make_follower(get_robot_model(REBOT_MIT_FF), "/dev/ttyACM0", "arm")
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
    assert robot.config.max_relative_target["elbow_flex"] == 20.0


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
    assert (kp, kd) == (45.0, 12.0)  # LeRobot's gains: the loop is no stiffer
    assert torque == pytest.approx(0.0, abs=1e-9)  # a vertical axis carries no weight
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


def test_plain_mit_model_carries_the_arms_weight_but_sends_no_velocity() -> None:
    robot = make_follower(get_robot_model(REBOT_MIT), "/dev/ttyACM0", "arm")
    assert type(robot) is RebotB601FeedforwardFollower
    assert robot.config.control_mode.value == "mit"
    assert robot.config.max_relative_target["shoulder_pan"] == 20.0
    assert robot.config.velocity_feedforward == 0.0
    assert robot.config.gravity_compensation == 1.0


WORK_POSE = {
    "shoulder_pan": -36.0,
    "shoulder_lift": -80.0,
    "elbow_flex": -48.0,
    "wrist_flex": 45.0,
    "wrist_yaw": -42.0,
    "wrist_roll": -74.0,
    "gripper": 0.0,
}


def _hold(robot, send, seconds: float) -> dict[str, tuple]:
    """Hold WORK_POSE, the arm already there, for `seconds`; returns the last commands."""
    for name, deg in WORK_POSE.items():
        robot.motors[name].position_rad = math.radians(deg)
    sent = {}
    for _ in range(round(seconds / 0.005)):
        sent = send(WORK_POSE)
    return sent


def test_gravity_torque_is_fed_forward_once_faded_in(follower) -> None:
    from lelab.rebot_gravity import gravity_torques

    robot, send = follower
    sent = _hold(robot, send, 1.5)
    model = gravity_torques([math.radians(WORK_POSE[j]) for j in ARM_JOINTS])

    torque = {joint: sent[joint][5] for joint in ARM_JOINTS}
    assert torque["shoulder_lift"] == pytest.approx(model[1])  # about +4.3 N.m here
    assert torque["elbow_flex"] == pytest.approx(model[2])  # about -5.8 N.m
    assert torque["wrist_flex"] == pytest.approx(1.10 * model[3])  # the scale tuned on the arm
    assert abs(torque["shoulder_lift"]) > 3.0 and abs(torque["elbow_flex"]) > 4.0
    assert torque["shoulder_pan"] == pytest.approx(0.0, abs=1e-9)  # a vertical axis carries no weight
    # Nothing else about the command changes: same gains, the arm is where it is asked to be.
    assert sent["elbow_flex"][3:5] == (45.0, 12.0)
    assert sent["gripper"][0] == "force_pos"


def test_gravity_torque_fades_in_from_the_first_command(follower) -> None:
    robot, send = follower
    first = _hold(robot, send, 0.005)
    assert first["elbow_flex"][5] == 0.0  # no jolt on a resting arm
    halfway = _hold(robot, send, 0.495)
    full = _hold(robot, send, 1.0)
    assert halfway["elbow_flex"][5] == pytest.approx(0.5 * full["elbow_flex"][5], rel=0.02)


def test_gravity_compensation_can_be_switched_off(follower) -> None:
    robot, send = follower
    robot.config.gravity_compensation = 0.0
    sent = _hold(robot, send, 1.5)
    assert all(sent[joint][5] == 0.0 for joint in ARM_JOINTS)


def test_gravity_model_matches_the_urdf_reference_values() -> None:
    """Values computed with pinocchio from Seeed's ReBot_Arm_DM.urdf for the same poses."""
    from lelab.rebot_gravity import gravity_torques

    rest = gravity_torques([0.0] * 6)
    assert rest == pytest.approx([0.0, -1.284, -7.183, -1.980, 0.0, 0.0], abs=2e-3)
    work = gravity_torques([math.radians(WORK_POSE[j]) for j in ARM_JOINTS])
    assert work == pytest.approx([0.0, 4.344, -5.775, -0.634, -0.463, 0.0], abs=2e-3)
    # Turning the base changes nothing: its axis is vertical.
    turned = gravity_torques([1.0, *[math.radians(WORK_POSE[j]) for j in ARM_JOINTS[1:]]])
    assert turned == pytest.approx(work, abs=1e-9)


def test_a_motor_that_shut_itself_down_stops_the_session_with_its_reason(follower) -> None:
    from lelab.utils.devices import MotorFaultError

    robot, send = follower
    rest = dict.fromkeys([*ARM_JOINTS, "gripper"], 0.0)
    send(rest)  # healthy motors: nothing raised

    healthy = robot.motors["elbow_flex"].get_state
    robot.motors["elbow_flex"].get_state = lambda: SimpleNamespace(
        pos=0.0, vel=0.0, torq=0.0, status_code=0x8, t_mos=41.0, t_rotor=55.0
    )
    with pytest.raises(MotorFaultError, match="elbow_flex reports over-voltage") as error:
        send(rest)
    assert "41/55" in str(error.value)  # temperatures, to tell a hot motor from a supply problem
    assert robot.bus.sent == []  # nothing is commanded to an arm that cannot hold itself

    robot.motors["elbow_flex"].get_state = healthy
    send(rest)


def test_a_fault_is_noticed_in_position_velocity_mode_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The position-velocity models never read the arm to send a command, only to
    observe it: the fault has to surface there, and through the teleoperation display."""
    from lelab.robot_models import REBOT_HIGH_POS_GAINS
    from lelab.teleoperate import get_joint_positions_from_robot
    from lelab.utils.devices import MotorFaultError

    robot = make_follower(get_robot_model(REBOT_HIGH_POS_GAINS), "/dev/ttyACM0", "arm")
    assert type(robot) is RebotB601FeedforwardFollower
    bus = _Bus()
    robot.bus = bus
    robot.motors = {name: _Motor(bus, name) for name in robot.motor_names}
    assert set(get_joint_positions_from_robot(robot, REBOT_HIGH_POS_GAINS)) == set(robot.motor_names)

    robot.motors["shoulder_lift"].get_state = lambda: SimpleNamespace(
        pos=0.0, vel=0.0, torq=0.0, status_code=0x9, t_mos=30.0, t_rotor=30.0
    )
    with pytest.raises(MotorFaultError, match="shoulder_lift reports under-voltage"):
        get_joint_positions_from_robot(robot, REBOT_HIGH_POS_GAINS)


def test_one_lost_feedback_reply_does_not_end_the_session(follower) -> None:
    robot, send = follower
    rest = dict.fromkeys([*ARM_JOINTS, "gripper"], 0.0)
    real = robot.motors["wrist_yaw"].get_state
    replies = iter([None, None])  # two reads in a row come back without this motor

    robot.motors["wrist_yaw"].get_state = lambda: next(replies, real())
    assert set(send(rest)) == {*ARM_JOINTS, "gripper"}  # the third read succeeded; every joint was commanded

    robot.motors["wrist_yaw"].get_state = lambda: None  # a motor that is really gone still stops it
    with pytest.raises(RuntimeError, match="No motor feedback available for: wrist_yaw"):
        send(rest)


def test_target_clipping_warnings_are_rate_limited(follower, caplog: pytest.LogCaptureFixture) -> None:
    """A fast move clips the target every cycle; LeRobot's warning for each must not flood the log."""
    import logging

    from lelab.rebot_ff_follower import _OncePerSecond

    robot, send = follower
    rest = dict.fromkeys([*ARM_JOINTS, "gripper"], 0.0)
    (limiter,) = [
        f for f in logging.getLogger("lerobot.robots.utils").filters if isinstance(f, _OncePerSecond)
    ]
    limiter._last = float("-inf")  # whatever earlier tests logged is not this test's second
    with caplog.at_level("WARNING", logger="lerobot.robots.utils"):
        for _ in range(100):  # half a second with the leader 60 degrees ahead of the arm
            send({**rest, "shoulder_pan": 60.0})
    clipped = [r for r in caplog.records if "Relative goal position magnitude" in r.getMessage()]
    assert 1 <= len(clipped) <= 2


def test_gravity_compensation_refuses_the_robstride_build() -> None:
    from lelab.rebot_ff_follower import RebotB601FeedforwardFollowerConfig as Config

    with pytest.raises(ValueError, match="Damiao"):
        RebotB601FeedforwardFollower(Config(port="can0", id="arm", motor_family="rs"))


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
            robot_model=REBOT_MIT_FF,
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
    assert cfg.robot.max_relative_target["wrist_flex"] == 20.0
    assert cfg.robot.max_relative_target["gripper"] == 270.0
    assert cfg.robot.velocity_feedforward == 1.0
