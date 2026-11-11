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
"""Tests for the robot models LeLab can drive, and the reBot B601 paths through each feature."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lelab.robot_models import (
    REBOT_DEFAULT,
    REBOT_HIGH_POS_GAINS,
    REBOT_MIT,
    ROBOT_MODELS,
    SO101,
    connect_arms,
    follower_cli_args,
    get_robot_model,
    make_follower_config,
    make_leader_config,
)


def test_get_robot_model_defaults_to_so101() -> None:
    assert get_robot_model(None).key == SO101
    assert get_robot_model("").key == SO101


def test_get_robot_model_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown robot model"):
        get_robot_model("not_a_robot")


@pytest.mark.parametrize("key", list(ROBOT_MODELS))
def test_model_names_match_the_lerobot_devices(key: str) -> None:
    """The registry repeats names LeRobot owns: the registered config type, and the
    device name that decides the calibration directory and a dataset's robot_type."""
    from lerobot.robots import RobotConfig
    from lerobot.teleoperators import TeleoperatorConfig

    model = ROBOT_MODELS[key]
    follower_config = make_follower_config(model, "/dev/ttyACM0", "arm")
    leader_config = make_leader_config(model, "/dev/ttyUSB0", "arm")

    # SO101FollowerConfig is also registered as so100_follower, so `.type` is not checked.
    assert RobotConfig.get_choice_class(model.follower_type) is type(follower_config)
    assert TeleoperatorConfig.get_choice_class(model.leader_type) is type(leader_config)
    assert follower_config.calibration_dir is None  # LeRobot derives it from the device name


@pytest.mark.parametrize("key", list(ROBOT_MODELS))
def test_model_names_match_the_lerobot_device_classes(key: str) -> None:
    import importlib

    model = ROBOT_MODELS[key]
    modules = {
        "so101": (
            "lerobot.robots.so_follower",
            "SO101Follower",
            "lerobot.teleoperators.so_leader",
            "SO101Leader",
        ),
        "rebot_b601": (
            "lerobot.robots.rebot_b601_follower",
            "RebotB601Follower",
            "lerobot.teleoperators.rebot_102_leader",
            "RebotArm102Leader",
        ),
    }[model.family]
    follower_cls = getattr(importlib.import_module(modules[0]), modules[1])
    leader_cls = getattr(importlib.import_module(modules[2]), modules[3])
    assert follower_cls.name == model.follower_name
    assert leader_cls.name == model.leader_name


def test_rebot_variants_differ_only_in_how_the_follower_is_driven() -> None:
    from lelab.rebot_tuning import FACTORY_POSITION_KP, HIGH_POSITION_KP

    default, high, mit = (get_robot_model(k) for k in (REBOT_DEFAULT, REBOT_HIGH_POS_GAINS, REBOT_MIT))

    # Same hardware: same drivers, calibration files and dataset robot_type.
    for model in (default, high, mit):
        assert (model.family, model.follower_name, model.leader_name) == (
            "rebot_b601",
            "rebot_b601_follower",
            "rebot_102_leader",
        )

    modes = [make_follower_config(m, "/dev/ttyACM0", "arm").control_mode.value for m in (default, high, mit)]
    assert modes == ["pos_vel", "pos_vel", "mit"]
    assert follower_cli_args(high) == ["--robot.control_mode=pos_vel"]
    assert follower_cli_args(mit) == ["--robot.control_mode=mit"]
    assert follower_cli_args(get_robot_model(SO101)) == []

    assert default.position_kp == FACTORY_POSITION_KP
    assert high.position_kp == HIGH_POSITION_KP
    assert mit.position_kp is None  # MIT mode does not use the motor's position loop


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # The default variant writes the factory gains: a raised gain outlives the session that set it.
        (REBOT_DEFAULT, {"shoulder_pan": 54.0, "wrist_roll": 54.0}),
        (REBOT_HIGH_POS_GAINS, {"shoulder_lift": 180.0, "wrist_flex": 100.0}),
    ],
)
def test_prepare_follower_writes_the_variants_gains(key: str, expected: dict, gains_log: list) -> None:
    from lelab.robot_models import prepare_follower

    prepare_follower(get_robot_model(key), "/dev/ttyACM0")
    ((port, gains),) = gains_log
    assert port == "/dev/ttyACM0"
    assert {joint: gains[joint] for joint in expected} == expected
    assert "gripper" not in gains


def test_prepare_follower_leaves_mit_and_so101_motors_alone(gains_log: list) -> None:
    from lelab.robot_models import prepare_follower

    prepare_follower(get_robot_model(REBOT_MIT), "/dev/ttyACM0")
    prepare_follower(get_robot_model(SO101), "/dev/ttyACM0")
    assert gains_log == []


@pytest.fixture(autouse=True)
def gains_log(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """No test here opens a real motor bus: record the (port, gains) that would be applied."""
    import lelab.rebot_tuning as tuning

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        tuning, "apply_position_gains", lambda port, gains, baud=921600: calls.append((port, gains))
    )
    return calls


class _Recorder:
    def __init__(self, log: list[str], name: str, fail: bool = False) -> None:
        self._log, self._name, self._fail = log, name, fail
        self.cameras: dict = {}
        self.config = SimpleNamespace(port="/dev/ttyACM0")

    def connect(self, calibrate: bool = True) -> None:
        self._log.append(f"{self._name}.connect(calibrate={calibrate})")
        if self._fail:
            raise OSError("no such port")

    @property
    def bus(self):
        raise AssertionError("a zero-calibrated arm is brought up through connect(), not its bus")


def test_connect_arms_connects_a_rebot_leader_before_powering_the_follower() -> None:
    log: list[str] = []
    connect_arms(get_robot_model(REBOT_HIGH_POS_GAINS), _Recorder(log, "follower"), _Recorder(log, "leader"))
    assert log == ["leader.connect(calibrate=False)", "follower.connect(calibrate=False)"]


def test_connect_arms_sets_the_rebot_gains_before_the_follower_connects(gains_log: list) -> None:
    log: list[str] = []

    class _Follower(_Recorder):
        def connect(self, calibrate: bool = True) -> None:
            # By now the gains are in the motors, written on a bus that is closed again.
            assert [port for port, _ in gains_log] == ["/dev/ttyACM0"]
            super().connect(calibrate)

    connect_arms(
        get_robot_model(REBOT_HIGH_POS_GAINS),
        _Follower(log, "follower"),
        _Recorder(log, "leader"),
        follower_port="/dev/ttyACM0",
    )
    assert log == ["leader.connect(calibrate=False)", "follower.connect(calibrate=False)"]


def test_connect_arms_reports_the_follower_when_its_gains_cannot_be_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.rebot_tuning as tuning

    def fail(port, gains, baud=921600):
        raise RuntimeError("shoulder_pan: no acknowledgement")

    monkeypatch.setattr(tuning, "apply_position_gains", fail)
    log: list[str] = []
    with pytest.raises(RuntimeError, match="follower arm on /dev/ttyACM0"):
        connect_arms(
            get_robot_model(REBOT_HIGH_POS_GAINS), _Recorder(log, "follower"), follower_port="/dev/ttyACM0"
        )
    assert log == []  # the follower was never powered


class _FakeMotor:
    def __init__(self, bus, joint_id: int) -> None:
        self.bus, self.joint_id, self.kp, self.closed = bus, joint_id, 54.0, False

    def write_register_f32(self, rid: int, value: float) -> None:
        self.bus.writes.append((self.joint_id, rid, value))
        if self.bus.missed_acks > 0:
            self.bus.missed_acks -= 1
            raise RuntimeError("write ack not received")
        if not self.bus.ignore_writes:
            self.kp = value

    def get_register_f32(self, rid: int, timeout_ms: int = 1000) -> float:
        return self.kp

    def store_parameters(self) -> None:
        self.bus.stored_to_flash = True

    def close(self) -> None:
        self.closed = True


class _FakeBus:
    def __init__(self, missed_acks: int = 0, ignore_writes: bool = False) -> None:
        self.writes: list[tuple[int, int, float]] = []
        self.motors: list[_FakeMotor] = []
        self.missed_acks, self.ignore_writes, self.closed = missed_acks, ignore_writes, False
        self.stored_to_flash = False

    def add_damiao_motor(self, send_id: int, recv_id: int, model: str) -> _FakeMotor:
        self.motors.append(_FakeMotor(self, send_id))
        return self.motors[-1]

    def close(self) -> None:
        self.closed = True


def _real_apply_position_gains(monkeypatch: pytest.MonkeyPatch, bus: _FakeBus):
    """The real function (the autouse fixture stubs the module attribute), on a fake bus."""
    import importlib

    import lelab.rebot_tuning as tuning

    monkeypatch.undo()
    tuning = importlib.reload(tuning)
    monkeypatch.setattr(tuning, "_open_bus", lambda port, baud: bus)
    monkeypatch.setattr(tuning.time, "sleep", lambda s: None)
    return tuning.apply_position_gains, tuning.HIGH_POSITION_KP


def test_position_gains_are_written_to_ram_per_motor_and_the_bus_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _FakeBus(missed_acks=2)  # the first two writes lose their acknowledgement
    apply, gains = _real_apply_position_gains(monkeypatch, bus)
    apply("/dev/ttyACM0", gains)

    stored = {motor.joint_id: motor.kp for motor in bus.motors}
    # CAN ids 1-3 are the 4340P joints, 4-6 the 4310 wrist; the gripper (7) is not touched.
    assert stored == {1: 270.0, 2: 180.0, 3: 200.0, 4: 100.0, 5: 100.0, 6: 100.0}
    assert {rid for _, rid, _ in bus.writes} == {27}  # only KP_APR
    assert bus.stored_to_flash is False
    assert bus.closed and all(motor.closed for motor in bus.motors)


def test_position_gains_fail_loudly_when_a_motor_does_not_take_them(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus(ignore_writes=True)
    with pytest.raises(RuntimeError, match="shoulder_pan.*KP_APR reads 54"):
        apply, gains = _real_apply_position_gains(monkeypatch, bus)
        apply("/dev/ttyACM0", gains)
    assert bus.closed  # the port is free for the next attempt


def test_connect_arms_never_powers_the_rebot_follower_when_the_leader_fails() -> None:
    log: list[str] = []
    with pytest.raises(RuntimeError, match="leader arm on /dev/ttyUSB0"):
        connect_arms(
            get_robot_model(REBOT_HIGH_POS_GAINS),
            _Recorder(log, "follower"),
            _Recorder(log, "leader", fail=True),
            leader_port="/dev/ttyUSB0",
        )
    assert log == ["leader.connect(calibrate=False)"]


def test_connect_arms_names_the_rebot_follower_port_on_failure() -> None:
    with pytest.raises(RuntimeError, match="follower arm on /dev/ttyACM0"):
        connect_arms(
            get_robot_model(REBOT_HIGH_POS_GAINS),
            _Recorder([], "follower", fail=True),
            follower_port="/dev/ttyACM0",
        )


# --- persisted state ---------------------------------------------------------


def test_calibration_dirs_follow_the_model(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    assert cfg.calibration_dirs() == (cfg.LEADER_CONFIG_PATH, cfg.FOLLOWER_CONFIG_PATH)
    leader_dir, follower_dir = cfg.calibration_dirs(REBOT_HIGH_POS_GAINS)
    assert Path(leader_dir).name == "rebot_102_leader"
    assert Path(follower_dir).name == "rebot_b601_follower"


def test_setup_calibration_files_reads_the_models_directories(tmp_lerobot_home: Path) -> None:
    from lelab.utils import config as cfg

    leader_dir, follower_dir = cfg.calibration_dirs(REBOT_HIGH_POS_GAINS)
    for directory, name in ((leader_dir, "lead.json"), (follower_dir, "follow.json")):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / name).write_text("{}")

    assert cfg.setup_calibration_files("lead.json", "follow.json", REBOT_HIGH_POS_GAINS) == ("lead", "follow")
    assert cfg.setup_follower_calibration_file("follow.json", REBOT_HIGH_POS_GAINS) == "follow"
    # The same names are not SO-101 calibrations.
    with pytest.raises(FileNotFoundError):
        cfg.setup_calibration_files("lead.json", "follow.json")


def test_robot_record_defaults_to_so101_and_reads_old_records(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.utils import config as cfg

    robots = tmp_lerobot_home / "robots"
    robots.mkdir(exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots))

    cfg.save_robot_record("fresh", {})
    assert cfg.get_robot_record("fresh")["robot_model"] == SO101

    # A record written before robot models existed.
    (robots / "old.json").write_text(json.dumps({"name": "old", "leader_port": "/dev/ttyUSB0"}))
    assert cfg.get_robot_record("old")["robot_model"] == SO101


def test_changing_a_records_model_unlinks_its_calibrations(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.utils import config as cfg

    robots = tmp_lerobot_home / "robots"
    robots.mkdir(exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots))

    cfg.save_robot_record(
        "arm",
        {
            "leader_port": "/dev/ttyUSB0",
            "follower_port": "/dev/ttyACM0",
            "leader_config": "arm.json",
            "follower_config": "arm.json",
        },
    )
    cfg.save_robot_record("arm", {"robot_model": REBOT_HIGH_POS_GAINS})
    record = cfg.get_robot_record("arm")
    assert record["robot_model"] == REBOT_HIGH_POS_GAINS
    assert (record["leader_config"], record["follower_config"]) == ("", "")
    assert record["leader_port"] == "/dev/ttyUSB0"  # ports survive

    # Re-sending the same model is not a change.
    cfg.save_robot_record("arm", {"robot_model": REBOT_HIGH_POS_GAINS, "follower_config": "f.json"})
    cfg.save_robot_record("arm", {"robot_model": REBOT_HIGH_POS_GAINS})
    assert cfg.get_robot_record("arm")["follower_config"] == "f.json"

    # Another way of driving the same arms uses the same calibration files.
    cfg.save_robot_record("arm", {"robot_model": REBOT_MIT})
    record = cfg.get_robot_record("arm")
    assert (record["robot_model"], record["follower_config"]) == (REBOT_MIT, "f.json")

    assert cfg.save_robot_record("arm", {"robot_model": "not_a_robot"}) is False
    assert cfg.get_robot_record("arm")["robot_model"] == REBOT_MIT


def test_record_is_clean_only_with_calibrations_of_its_own_model(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.utils import config as cfg

    leader_dir, follower_dir = cfg.calibration_dirs(REBOT_HIGH_POS_GAINS)
    for directory in (leader_dir, follower_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "arm.json").write_text("{}")

    record = {
        "name": "arm",
        "leader_port": "/dev/ttyUSB0",
        "follower_port": "/dev/ttyACM0",
        "leader_config": "arm.json",
        "follower_config": "arm.json",
    }
    assert cfg.is_robot_record_clean({**record, "robot_model": REBOT_HIGH_POS_GAINS}) is True
    assert cfg.is_robot_record_clean({**record, "robot_model": SO101}) is False


def test_robot_endpoints_carry_the_model(
    client, tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.utils import config as cfg

    robots = tmp_lerobot_home / "robots"
    robots.mkdir(exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots))

    models = client.get("/robot-models").json()
    assert {m["key"] for m in models["models"]} == set(ROBOT_MODELS)
    rebot = next(m for m in models["models"] if m["key"] == REBOT_HIGH_POS_GAINS)
    assert rebot["calibration"] == "zero"
    assert rebot["has_urdf"] is False
    assert rebot["dataset_robot_type"] == "rebot_b601_follower"

    created = client.post("/robots/arm?create=true", json={"robot_model": REBOT_HIGH_POS_GAINS})
    assert created.json()["robot"]["robot_model"] == REBOT_HIGH_POS_GAINS
    assert client.post("/robots/arm", json={"robot_model": "not_a_robot"}).status_code == 400

    _, follower_dir = cfg.calibration_dirs(REBOT_HIGH_POS_GAINS)
    Path(follower_dir).mkdir(parents=True, exist_ok=True)
    (Path(follower_dir) / "follower1.json").write_text("{}")
    listed = client.get(f"/calibration-configs/robot?robot_model={REBOT_HIGH_POS_GAINS}").json()
    assert [c["filename"] for c in listed["configs"]] == ["follower1.json"]
    assert client.get("/calibration-configs/robot").json()["configs"] == []
    assert client.get("/calibration-configs/robot?robot_model=not_a_robot").status_code == 422


# --- feature modules ---------------------------------------------------------


def test_joint_positions_of_a_model_without_urdf_keep_their_motor_names() -> None:
    from lelab.teleoperate import get_joint_positions_from_robot

    class _Robot:
        def get_observation(self):
            return {"shoulder_pan.pos": 90.0, "wrist_yaw.pos": -45.0, "front": object()}

    positions = get_joint_positions_from_robot(_Robot(), REBOT_HIGH_POS_GAINS)
    assert positions == {
        "shoulder_pan": pytest.approx(math.pi / 2),
        "wrist_yaw": pytest.approx(-math.pi / 4),
    }


def test_teleoperation_builds_the_requested_models_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    seen: dict = {}

    def setup(leader, follower, robot_model=None):
        seen["setup_model"] = robot_model
        return "leader", "follower"

    class _Device:
        def __init__(self, name: str) -> None:
            self.name = name

        def disconnect(self) -> None:
            seen.setdefault("disconnected", []).append(self.name)

    def connect(model, robot, teleop_device, **ports):
        seen["connect_model"] = model.key
        raise RuntimeError("stop before the loop")

    monkeypatch.setattr(teleop, "setup_calibration_files", setup)
    monkeypatch.setattr(
        teleop, "make_follower", lambda model, port, device_id: _Device(f"{model.key}-follower")
    )
    monkeypatch.setattr(teleop, "make_leader", lambda model, port, device_id: _Device(f"{model.key}-leader"))
    monkeypatch.setattr(teleop, "connect_arms", connect)

    result = teleop.handle_start_teleoperation(
        teleop.TeleoperateRequest(
            leader_port="/dev/ttyUSB0",
            follower_port="/dev/ttyACM0",
            leader_config="leader.json",
            follower_config="follower.json",
            robot_model=REBOT_HIGH_POS_GAINS,
        )
    )

    assert result == {"success": False, "message": "stop before the loop"}
    assert seen["setup_model"] == REBOT_HIGH_POS_GAINS
    assert seen["connect_model"] == REBOT_HIGH_POS_GAINS
    assert seen["disconnected"] == ["rebot_high_pos_gains-follower", "rebot_high_pos_gains-leader"]
    assert teleop.teleoperation_active is False


def test_requests_reject_an_unknown_robot_model() -> None:
    from pydantic import ValidationError

    from lelab.teleoperate import TeleoperateRequest

    with pytest.raises(ValidationError):
        TeleoperateRequest(
            leader_port="a", follower_port="b", leader_config="c", follower_config="d", robot_model="nope"
        )


def test_record_config_uses_the_requested_models_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab import record

    monkeypatch.setattr(
        record, "setup_calibration_files", lambda leader, follower, robot_model=None: ("leader", "follower")
    )
    cfg = record.create_record_config(
        record.RecordingRequest(
            leader_port="/dev/ttyUSB0",
            follower_port="/dev/ttyACM0",
            leader_config="leader.json",
            follower_config="follower.json",
            robot_model=REBOT_HIGH_POS_GAINS,
            dataset_repo_id="local/rebot_test",
            single_task="pick",
        )
    )
    assert cfg.robot.type == "rebot_b601_follower"
    assert cfg.teleop.type == "rebot_102_leader"
    assert cfg.robot.control_mode.value == "pos_vel"


def test_rebot_rollout_argv_parses(tmp_path, monkeypatch: pytest.MonkeyPatch, gains_log: list) -> None:
    """The rollout command LeLab builds for a reBot follower is one LeRobot's parser accepts."""
    from lelab import rollout
    from lelab.rollout import InferenceRequest
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts import lerobot_rollout

    checkpoint = tmp_path / "checkpoints" / "000050" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )._save_pretrained(checkpoint)

    commands: list[list[str]] = []
    seen_models: list = []

    def capture(cmd, **kwargs):
        commands.append(cmd)
        raise OSError("stop before spawning")

    def setup(name, robot_model=None):
        seen_models.append(robot_model)
        return "follower1"

    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(rollout, "setup_follower_calibration_file", setup)
    monkeypatch.setattr(rollout.subprocess, "Popen", capture)
    rollout.handle_start_inference(
        InferenceRequest(
            follower_port="/dev/ttyACM0",
            follower_config="follower1.json",
            robot_model=REBOT_HIGH_POS_GAINS,
            policy_ref=str(checkpoint),
            task="pick the cube",
        )
    )
    assert seen_models == [REBOT_HIGH_POS_GAINS]
    assert [port for port, _ in gains_log] == ["/dev/ttyACM0"]  # set before the subprocess takes the port

    parsed = []

    class _StopBeforeHardwareError(Exception):
        pass

    def stop_before_hardware(cfg, _shutdown_event):
        parsed.append(cfg)
        raise _StopBeforeHardwareError

    monkeypatch.setattr(lerobot_rollout, "build_rollout_context", stop_before_hardware)
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", *commands[0][3:]])
    with pytest.raises(_StopBeforeHardwareError):
        lerobot_rollout.main()

    (cfg,) = parsed
    assert cfg.robot.type == "rebot_b601_follower"
    assert cfg.robot.port == "/dev/ttyACM0"
    assert cfg.robot.id == "follower1"
    assert cfg.robot.control_mode.value == "pos_vel"


# --- zero-position calibration ----------------------------------------------


class _ZeroDevice:
    """A follower whose calibrate() is written for a terminal, like LeRobot's reBot drivers."""

    calibration_fpath = "/tmp/arm.json"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.prompts: list[str] = []
        self.calibration = {"shoulder_pan": object()}

    def connect(self, calibrate: bool = True) -> None:
        self.calls.append(f"connect(calibrate={calibrate})")

    def disable_torque(self) -> None:
        self.calls.append("disable_torque")

    def calibrate(self) -> None:
        if self.calibration:
            answer = input("Press ENTER to use the calibration file, or type 'c' to calibrate: ")
            self.prompts.append(answer)
            if answer.strip().lower() != "c":
                return
        input("Press ENTER when ready...")
        self.calls.append("set_zero")

    def disconnect(self) -> None:
        self.calls.append("disconnect")


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


def _start_zero_calibration(monkeypatch: pytest.MonkeyPatch, device: _ZeroDevice, robot_name=None):
    import lelab.calibrate as calibrate

    monkeypatch.setattr(calibrate, "make_robot_from_config", lambda config: device)
    mgr = calibrate.CalibrationManager()
    result = mgr.start_calibration(
        calibrate.CalibrationRequest(
            device_type="robot",
            port="/dev/ttyACM0",
            config_file="arm",
            robot_name=robot_name,
            robot_model=REBOT_HIGH_POS_GAINS,
        )
    )
    assert result["success"] is True
    _wait_for(lambda: mgr.get_status().status == "awaiting_zero")
    return mgr


def test_zero_calibration_waits_for_the_user_before_setting_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    device = _ZeroDevice()
    mgr = _start_zero_calibration(monkeypatch, device)

    # Torque is off so the arm can be posed, and nothing is zeroed yet.
    assert device.calls == ["connect(calibrate=False)", "disable_torque"]
    assert "zero position" in mgr.get_status().message

    assert mgr.complete_step()["success"] is True
    _wait_for(lambda: not mgr.get_status().calibration_active)

    assert mgr.get_status().status == "completed"
    assert device.prompts == ["c"]  # the terminal prompt was answered: recalibrate
    assert device.calls == ["connect(calibrate=False)", "disable_torque", "set_zero", "disconnect"]


def test_cancelling_zero_calibration_leaves_the_stored_zero_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    device = _ZeroDevice()
    mgr = _start_zero_calibration(monkeypatch, device)

    assert mgr.stop_calibration_process()["success"] is True

    assert "set_zero" not in device.calls
    assert device.calls[-1] == "disconnect"
    assert mgr.get_status().calibration_active is False


def test_zero_calibration_links_the_file_to_the_robot_record(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lelab.utils import config as cfg

    robots = tmp_lerobot_home / "robots"
    robots.mkdir(exist_ok=True)
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots))
    cfg.save_robot_record("arm", {"robot_model": REBOT_HIGH_POS_GAINS})

    mgr = _start_zero_calibration(monkeypatch, _ZeroDevice(), robot_name="arm")
    mgr.complete_step()
    _wait_for(lambda: not mgr.get_status().calibration_active)

    record = cfg.get_robot_record("arm")
    assert record["follower_config"] == "arm.json"
    assert record["follower_port"] == "/dev/ttyACM0"


def test_start_calibration_rejects_an_unknown_robot_model() -> None:
    from lelab.calibrate import CalibrationManager, CalibrationRequest

    result = CalibrationManager().start_calibration(
        CalibrationRequest(device_type="robot", port="/dev/ttyACM0", config_file="arm", robot_model="nope")
    )
    assert result["success"] is False
    assert "Unknown robot model" in result["message"]
