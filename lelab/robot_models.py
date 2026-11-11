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

"""The leader/follower arm pairs LeLab can drive.

Each feature module (teleoperation, recording, replay, calibration, rollout)
asks this module for the LeRobot configs and devices of the robot model named
in the request, and for how to bring them up, so none of them names a
hardware class itself.

LeRobot is imported inside the functions: this module is imported by
`utils/config.py`, which must stay cheap, and the reBot drivers need the
optional `rebot` extra.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from .rebot_tuning import FACTORY_POSITION_KP, HIGH_POSITION_KP

logger = logging.getLogger(__name__)

SO101 = "so101"
REBOT_DEFAULT = "rebot_default"
REBOT_HIGH_POS_GAINS = "rebot_high_pos_gains"
REBOT_MIT = "rebot_mit"
REBOT_MIT_FF = "rebot_mit_ff"
DEFAULT_ROBOT_MODEL = SO101

RobotModelLiteral = Literal["so101", "rebot_default", "rebot_high_pos_gains", "rebot_mit", "rebot_mit_ff"]


@dataclass(frozen=True)
class RobotModel:
    key: str
    label: str
    # The hardware: models of one family share drivers, calibration files and
    # dataset robot_type, and differ only in how the follower is driven.
    family: Literal["so101", "rebot_b601"]
    # `--robot.type` / `--teleop.type` names LeRobot registers the configs under.
    follower_type: str
    leader_type: str
    # `Robot.name` / `Teleoperator.name`: the calibration sub-directory LeRobot reads,
    # and the `robot_type` it stamps on datasets.
    follower_name: str
    leader_name: str
    # "range": record each joint's range of motion (Feetech).
    # "zero": pose the arm at its zero position, which the motors then store.
    calibration: Literal["range", "zero"]
    # Motor name -> joint name in the URDF the frontend renders, when there is one.
    urdf_joints: dict[str, str] | None
    # pip extra of LeLab that installs the drivers, when they are optional.
    extra: str | None = None
    # Follower config fields that differ from LeRobot's defaults; also sent as
    # `--robot.<name>=<value>` to LeRobot's CLIs.
    follower_options: dict[str, Any] = field(default_factory=dict)
    # Position-loop gain (KP_APR) to put in each motor before the follower is
    # connected; None leaves the motors alone. See rebot_tuning.py.
    position_kp: dict[str, float] | None = None


# How far, in degrees, a MIT-mode target may lead the arm. The gripper is driven in
# force-position mode with its own speed limit, so its bound is its whole travel.
_REBOT_MIT_OPTIONS: dict[str, Any] = {
    "control_mode": "mit",
    "max_relative_target": {
        "shoulder_pan": 20.0,
        "shoulder_lift": 20.0,
        "elbow_flex": 20.0,
        "wrist_flex": 20.0,
        "wrist_yaw": 20.0,
        "wrist_roll": 20.0,
        "gripper": 270.0,
    },
}

# LeLab's own follower for the B601, registered by lelab/rebot_ff_follower.py.
_REBOT_FEEDFORWARD_FOLLOWER = "lelab_rebot_b601_ff_follower"


def _rebot(
    key: str,
    label: str,
    follower_options: dict[str, Any],
    position_kp: dict[str, float] | None,
    follower_type: str = "rebot_b601_follower",
) -> RobotModel:
    return RobotModel(
        key=key,
        label=f"reBot B601 + Arm 102 leader: {label}",
        family="rebot_b601",
        follower_type=follower_type,
        leader_type="rebot_102_leader",
        follower_name="rebot_b601_follower",
        leader_name="rebot_102_leader",
        calibration="zero",
        urdf_joints=None,
        extra="rebot",
        follower_options=follower_options,
        position_kp=position_kp,
    )


ROBOT_MODELS: dict[str, RobotModel] = {
    SO101: RobotModel(
        key=SO101,
        label="SO-101",
        family="so101",
        follower_type="so101_follower",
        leader_type="so101_leader",
        follower_name="so_follower",
        leader_name="so_leader",
        calibration="range",
        urdf_joints={
            "shoulder_pan": "Rotation",
            "shoulder_lift": "Pitch",
            "elbow_flex": "Elbow",
            "wrist_flex": "Wrist_Pitch",
            "wrist_roll": "Wrist_Roll",
            "gripper": "Jaw",
        },
    ),
    # The same arms, driven four ways. Position-velocity mode closes the gap to
    # a target at a bounded speed (the motor profile's `pos_vel_velocity`), which
    # suits sessions started from a button with the follower wherever it was left.
    #
    # Factory gains: the follower trails the leader by 0.2 s (wrist) to 0.7 s
    # (shoulder, elbow).
    REBOT_DEFAULT: _rebot(
        REBOT_DEFAULT, "default (factory gains)", {"control_mode": "pos_vel"}, FACTORY_POSITION_KP
    ),
    # Raised position-loop gains, as far as the whole arm stays stable: roughly half the lag.
    REBOT_HIGH_POS_GAINS: _rebot(
        REBOT_HIGH_POS_GAINS, "high position gains", {"control_mode": "pos_vel"}, HIGH_POSITION_KP
    ),
    # MIT mode: a spring and damper to the target, with LeRobot's gains. It has no
    # speed limit of its own, so the target is kept within `max_relative_target`
    # degrees of the arm: the spring cannot wind up when a session starts with the
    # follower away from the leader's pose.
    #
    # The damper resists every motion, so the follower trails by about kd / kp
    # seconds. For imitation learning that is not a defect to tune away: a soft,
    # well-damped controller turns a policy's action errors into small state
    # errors, and tighter tracking fits the data better but performs worse in
    # closed loop (Bronars, Park, Agrawal, "Tune to Learn", arXiv:2604.02523).
    REBOT_MIT: _rebot(REBOT_MIT, "MIT mode", _REBOT_MIT_OPTIONS, None),
    # The same, with LeLab's follower also sending the target's velocity
    # (lelab/rebot_ff_follower.py): the lag goes without either gain going up, which
    # is what teleoperation wants. It is the tight-tracking regime that paper warns
    # against for learning, and the velocity is differentiated from the targets, so
    # a policy's jitter reaches the motors through the damping gain.
    REBOT_MIT_FF: _rebot(
        REBOT_MIT_FF,
        "MIT mode with velocity feed-forward",
        _REBOT_MIT_OPTIONS,
        None,
        follower_type=_REBOT_FEEDFORWARD_FOLLOWER,
    ),
}


def get_robot_model(key: str | None) -> RobotModel:
    """The model registered under `key` (the default model for None or "")."""
    model = ROBOT_MODELS.get(key or DEFAULT_ROBOT_MODEL)
    if model is None:
        raise ValueError(f"Unknown robot model {key!r}. Known models: {', '.join(ROBOT_MODELS)}")
    return model


def is_robot_model(key: Any) -> bool:
    return isinstance(key, str) and key in ROBOT_MODELS


def prepare_follower(model: RobotModel, port: str) -> None:
    """Settings a follower needs in its motors before anything connects to it.

    Runs on a connection of its own, so the port must be free: call it before
    the follower is connected, or before a subprocess that will connect to it.
    """
    if model.position_kp is not None:
        from .rebot_tuning import apply_position_gains

        apply_position_gains(port, model.position_kp)


def follower_cli_args(model: RobotModel) -> list[str]:
    """`--robot.*` flags, besides type/port/id/cameras, that LeRobot's CLIs need for this follower."""
    return [
        f"--robot.{name}={json.dumps(value) if isinstance(value, dict) else value}"
        for name, value in model.follower_options.items()
    ]


def _missing_extra(model: RobotModel, error: ImportError) -> ImportError:
    return ImportError(
        f"The {model.label} drivers are not installed ({error}). "
        f'Install them with: pip install "lelab[{model.extra}]"'
    )


def make_follower_config(model: RobotModel, port: str, device_id: str, cameras: dict | None = None):
    cameras = cameras or {}
    if model.family == "so101":
        from lerobot.robots.so_follower import SO101FollowerConfig

        return SO101FollowerConfig(port=port, id=device_id, cameras=cameras, **model.follower_options)
    if model.family == "rebot_b601":
        if model.follower_type == _REBOT_FEEDFORWARD_FOLLOWER:
            from .rebot_ff_follower import RebotB601FeedforwardFollowerConfig as FollowerConfig
        else:
            from lerobot.robots.rebot_b601_follower import RebotB601FollowerRobotConfig as FollowerConfig

        return FollowerConfig(port=port, id=device_id, cameras=cameras, **model.follower_options)
    raise ValueError(f"Unknown robot model {model.key!r}")


def make_leader_config(model: RobotModel, port: str, device_id: str):
    if model.family == "so101":
        from lerobot.teleoperators.so_leader import SO101LeaderConfig

        return SO101LeaderConfig(port=port, id=device_id)
    if model.family == "rebot_b601":
        from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderTeleopConfig

        return RebotArm102LeaderTeleopConfig(port=port, id=device_id)
    raise ValueError(f"Unknown robot model {model.key!r}")


def make_follower(model: RobotModel, port: str, device_id: str, cameras: dict | None = None):
    """An unconnected follower arm."""
    from lerobot.robots import make_robot_from_config

    try:
        return make_robot_from_config(make_follower_config(model, port, device_id, cameras))
    except ImportError as e:
        if model.extra is None:
            raise
        raise _missing_extra(model, e) from e


def make_leader(model: RobotModel, port: str, device_id: str):
    """An unconnected leader arm."""
    from lerobot.teleoperators import make_teleoperator_from_config

    try:
        return make_teleoperator_from_config(make_leader_config(model, port, device_id))
    except ImportError as e:
        if model.extra is None:
            raise
        raise _missing_extra(model, e) from e


def _connection_error(side: str, port: str | None, cause: Exception) -> RuntimeError:
    where = f" on {port}" if port else ""
    logger.error(f"{side.capitalize()} arm connection failed: {cause}")
    return RuntimeError(
        f"Could not connect to the {side} arm{where}. "
        "Make sure it's plugged in and powered on, then try again."
    )


def connect_arms(
    model: RobotModel,
    robot,
    teleop=None,
    *,
    follower_port: str | None = None,
    leader_port: str | None = None,
) -> None:
    """Bring up the follower, and the leader when there is one, ready to be driven.

    A connection failure is raised as a RuntimeError naming the arm and its
    port. The caller owns the cleanup: on any error, disconnect both devices.
    """
    if model.calibration == "range":
        _connect_feetech_arms(robot, teleop, follower_port, leader_port)
    else:
        _connect_zeroed_arms(model, robot, teleop, follower_port, leader_port)


def _connect_feetech_arms(robot, teleop, follower_port, leader_port) -> None:
    # `robot.connect()` can't be used here. It prompts on stdin when the motors
    # disagree with the calibration file (EOFError under the server), and it
    # ends with configure(), which leaves torque enabled — writing the
    # calibration registers afterwards corrupts the bus a few seconds in.
    logger.info("Connecting to follower arm...")
    try:
        robot.bus.connect()
    except Exception as e:
        raise _connection_error("follower", follower_port, e) from e

    if teleop is not None:
        logger.info("Connecting to leader arm...")
        try:
            teleop.bus.connect()
        except Exception as e:
            raise _connection_error("leader", leader_port, e) from e

    # Feetech motors only store calibration registers with the torque off,
    # which also opens their EEPROM lock.
    logger.info("Writing calibration to motors...")
    robot.bus.disable_torque()
    robot.bus.write_calibration(robot.calibration)
    if teleop is not None:
        teleop.bus.disable_torque()
        teleop.bus.write_calibration(teleop.calibration)

    logger.info("Connecting cameras and configuring motors...")
    for cam in robot.cameras.values():
        cam.connect()
    robot.configure()
    if teleop is not None:
        teleop.configure()


def _connect_zeroed_arms(model: RobotModel, robot, teleop, follower_port, leader_port) -> None:
    # These arms keep their zero in the motors, so there is nothing to write:
    # the calibration file only records that the zero was set. `calibrate=False`
    # keeps connect() from prompting on stdin.
    #
    # The leader goes first. Connecting the follower enables its torque, and a
    # leader that then fails to connect would have us release a powered arm.
    if teleop is not None:
        logger.info("Connecting to leader arm...")
        try:
            teleop.connect(calibrate=False)
        except Exception as e:
            raise _connection_error("leader", leader_port, e) from e

    logger.info("Connecting to follower arm...")
    try:
        prepare_follower(model, follower_port or robot.config.port)
        robot.connect(calibrate=False)
    except Exception as e:
        raise _connection_error("follower", follower_port, e) from e
