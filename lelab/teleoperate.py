# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import math
import threading
import time
from typing import Any

from pydantic import BaseModel

from .robot_models import (
    DEFAULT_ROBOT_MODEL,
    RobotModelLiteral,
    connect_arms,
    get_robot_model,
    make_follower,
    make_leader,
)
from .utils.config import setup_calibration_files
from .utils.devices import MotorFaultError, follower_lock, safe_disconnect_device

logger = logging.getLogger(__name__)

# Global variables for teleoperation state
teleoperation_active = False
teleoperation_thread: threading.Thread | None = None
current_robot = None
current_teleop = None
# Model of the arms in the active session; decides how joints map to the URDF.
current_robot_model: str = DEFAULT_ROBOT_MODEL
# Why the last session ended on its own, if it did; cleared when the next one starts.
last_error: str | None = None
# Timing of the last window of control cycles, for /teleoperation-status.
loop_stats: dict[str, float] | None = None

_LOOP_STATS_WINDOW_S = 2.0


class LoopTimer:
    """Averages the control loop's timing over fixed windows.

    The loop has no target rate: it runs as fast as the leader can be read and
    the follower commanded, so the rate it achieves is worth reporting.
    """

    def __init__(self, window_s: float = _LOOP_STATS_WINDOW_S, clock=time.perf_counter) -> None:
        self._window_s = window_s
        self._clock = clock
        self._window_start = clock()
        self._cycles = 0
        self._read_s = 0.0
        self._send_s = 0.0
        self._max_period_s = 0.0
        self._last_cycle_end: float | None = None

    def record(self, read_s: float, send_s: float) -> dict[str, float] | None:
        """Count one cycle; returns the window's stats when a window closes."""
        now = self._clock()
        if self._last_cycle_end is not None:
            self._max_period_s = max(self._max_period_s, now - self._last_cycle_end)
        self._last_cycle_end = now
        self._cycles += 1
        self._read_s += read_s
        self._send_s += send_s

        elapsed = now - self._window_start
        if elapsed < self._window_s:
            return None
        stats = {
            "hz": round(self._cycles / elapsed, 1),
            "leader_read_ms": round(1e3 * self._read_s / self._cycles, 2),
            "follower_send_ms": round(1e3 * self._send_s / self._cycles, 2),
            "max_period_ms": round(1e3 * self._max_period_s, 2),
        }
        self._window_start = now
        self._cycles = 0
        self._read_s = self._send_s = self._max_period_s = 0.0
        return stats


class TeleoperateRequest(BaseModel):
    leader_port: str
    follower_port: str
    leader_config: str
    follower_config: str
    robot_model: RobotModelLiteral = DEFAULT_ROBOT_MODEL


def get_joint_positions_from_robot(robot, robot_model: str = DEFAULT_ROBOT_MODEL) -> dict[str, float]:
    """
    Extract current joint positions from the robot and convert to URDF joint format.

    lerobot drives the SO-101 with ``use_degrees=True`` by default, so each
    ``observation["<motor>.pos"]`` is already the joint angle in degrees relative
    to the calibration center — which is also the URDF's zero pose. The URDF
    joint value is therefore just that angle converted to radians, for every
    joint. (The gripper reports 0–100 rather than degrees, but that range lands
    inside the Jaw limit, matching the open/closed sweep.)

    A model without a URDF in the frontend reports its joints under their motor
    names, still in radians.

    Args:
        robot: The connected follower
        robot_model: Key of the robot model the follower belongs to

    Returns:
        Dictionary mapping URDF joint names to radian values
    """
    motor_to_urdf_mapping = get_robot_model(robot_model).urdf_joints

    try:
        observation = robot.get_observation()
        if motor_to_urdf_mapping is None:
            return {
                key.removesuffix(".pos"): value * math.pi / 180.0
                for key, value in observation.items()
                if key.endswith(".pos")
            }

        joint_positions: dict[str, float] = {}
        debug_rows = []
        for motor_name, urdf_joint_name in motor_to_urdf_mapping.items():
            motor_key = f"{motor_name}.pos"
            if motor_key not in observation:
                logger.warning(f"Motor {motor_key} not found in observation")
                joint_positions[urdf_joint_name] = 0.0
                continue

            angle_degrees = observation[motor_key]
            joint_positions[urdf_joint_name] = angle_degrees * math.pi / 180.0
            debug_rows.append(f"{motor_name:14s} {angle_degrees:+8.2f}° → {urdf_joint_name:11s}")

        # Throttled debug print (~once per second at 20 Hz broadcast).
        now = time.time()
        if now - getattr(get_joint_positions_from_robot, "_last_log", 0) > 1.0:
            get_joint_positions_from_robot._last_log = now
            logger.info("[joint-debug]\n  " + "\n  ".join(debug_rows))

        return joint_positions

    except MotorFaultError:
        raise  # not a display hiccup: the arm is limp and the session has to end
    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return dict.fromkeys((motor_to_urdf_mapping or {}).values(), 0.0)


def _safe_disconnect(device) -> None:
    """Disconnect a robot/teleop device, swallowing (but logging) any error.

    Used on the connection-failure cleanup path so one device's failure can't
    leave the other holding its serial port open.
    """
    safe_disconnect_device(device, logger)


def handle_start_teleoperation(request: TeleoperateRequest, websocket_manager=None) -> dict[str, Any]:
    """Handle start teleoperation request.

    Connects to both arms *synchronously* so that a connection failure (arm
    unplugged, port busy, power off) is reported back to the caller, rather than
    dying silently in the worker thread while the API has already claimed
    success. Only the teleoperation loop runs in the background thread.
    """
    global teleoperation_active, teleoperation_thread, current_robot, current_teleop, current_robot_model
    global last_error

    from . import record as _record, replay as _replay, rollout as _rollout

    with follower_lock:
        if teleoperation_active:
            return {"success": False, "message": "Teleoperation is already active"}
        if _record.recording_active:
            return {"success": False, "message": "Recording is currently active. Stop it first."}
        if _rollout.inference_active:
            return {"success": False, "message": "Inference is currently active. Stop it first."}
        if _replay.replay_active:
            return {"success": False, "message": "Replay is currently active. Stop it first."}
        teleoperation_active = True
        last_error = None

    robot = None
    teleop_device = None
    try:
        logger.info(
            f"Starting teleoperation with leader port: {request.leader_port}, follower port: {request.follower_port}"
        )

        model = get_robot_model(request.robot_model)

        # Setup calibration files
        leader_config_name, follower_config_name = setup_calibration_files(
            request.leader_config, request.follower_config, model.key
        )

        # Connect synchronously. If either device fails to connect, clean up the
        # other (so its serial port is released) and report the error — do NOT
        # leave the caller thinking teleoperation started. The error names which
        # arm failed and tells the user what to do.
        logger.info("Initializing robot and teleop device...")
        robot = make_follower(model, request.follower_port, follower_config_name)
        teleop_device = make_leader(model, request.leader_port, leader_config_name)
        connect_arms(
            model,
            robot,
            teleop_device,
            follower_port=request.follower_port,
            leader_port=request.leader_port,
        )
        logger.info("Successfully connected to both devices")

        current_robot = robot
        current_teleop = teleop_device
        current_robot_model = model.key

        # Stream the arms in the background; the worker owns disconnect so stop()
        # does not race the serial bus from the request thread.
        def teleoperation_worker():
            global teleoperation_active, current_robot, current_teleop, loop_stats, last_error

            logger.info("Starting teleoperation loop...")
            try:
                last_broadcast_time = 0
                broadcast_interval = 1 / 30  # 30 FPS
                timer = LoopTimer()
                last_stats_log = 0.0

                while teleoperation_active:
                    cycle_start = time.perf_counter()
                    action = teleop_device.get_action()
                    read_done = time.perf_counter()
                    robot.send_action(action)
                    stats = timer.record(read_done - cycle_start, time.perf_counter() - read_done)
                    if stats is not None:
                        loop_stats = stats
                        if read_done - last_stats_log >= 10.0:
                            last_stats_log = read_done
                            logger.info(f"[teleop-loop] {stats}")

                    current_time = time.time()
                    if current_time - last_broadcast_time >= broadcast_interval:
                        try:
                            joint_positions = get_joint_positions_from_robot(robot, model.key)
                            joint_data = {
                                "type": "joint_update",
                                "joints": joint_positions,
                                "timestamp": current_time,
                            }
                            if websocket_manager and websocket_manager.active_connections:
                                websocket_manager.broadcast_joint_data_sync(joint_data)
                            last_broadcast_time = current_time
                        except MotorFaultError:
                            raise
                        except Exception as e:
                            logger.error(f"Error broadcasting joint data: {e}")

                    time.sleep(0.001)
            except Exception as e:
                last_error = str(e)
                logger.exception(f"Teleoperation stopped by an error: {e}")
            finally:
                _safe_disconnect(robot)
                _safe_disconnect(teleop_device)
                logger.info("Teleoperation stopped")
                teleoperation_active = False
                current_robot = None
                current_teleop = None
                loop_stats = None

        teleoperation_thread = threading.Thread(
            target=teleoperation_worker, name="teleoperation-worker", daemon=True
        )
        teleoperation_thread.start()

        return {
            "success": True,
            "message": "Teleoperation started successfully",
            "leader_port": request.leader_port,
            "follower_port": request.follower_port,
        }

    except Exception as e:
        # Connection (or setup) failed before the loop started: release any
        # device that did open, reset state, and surface the error.
        _safe_disconnect(robot)
        _safe_disconnect(teleop_device)
        teleoperation_active = False
        current_robot = None
        current_teleop = None
        logger.error(f"Failed to start teleoperation: {e}")
        # str(e) is already a user-facing message for the connection failures
        # raised above; the toast title supplies the "error starting" context.
        return {"success": False, "message": str(e)}


def handle_stop_teleoperation() -> dict[str, Any]:
    """Handle stop teleoperation request.

    Signals the worker via `teleoperation_active = False` and waits for it to
    exit. The worker owns the disconnect call, so this avoids racing the
    serial bus from the request thread.
    """
    global teleoperation_active, teleoperation_thread

    if not teleoperation_active:
        return {"success": False, "message": "No teleoperation session is active"}

    logger.info("Stop teleoperation triggered from web interface")
    teleoperation_active = False

    worker = teleoperation_thread
    if worker is not None and worker.is_alive():
        worker.join(timeout=5.0)
        if worker.is_alive():
            logger.warning("Teleoperation worker did not exit within 5s")
    teleoperation_thread = None

    return {"success": True, "message": "Teleoperation stopped successfully"}


def handle_teleoperation_status() -> dict[str, Any]:
    """Handle teleoperation status request"""
    return {
        "teleoperation_active": teleoperation_active,
        "robot_model": current_robot_model,
        # Achieved control-loop timing over the last window; None until one has elapsed.
        "loop": loop_stats if teleoperation_active else None,
        "last_error": last_error,
        "available_controls": {
            "stop_teleoperation": teleoperation_active,
        },
        "message": "Teleoperation status retrieved successfully",
    }


def handle_get_joint_positions() -> dict[str, Any]:
    """Handle get current robot joint positions request"""
    global current_robot

    if not teleoperation_active or current_robot is None:
        return {"success": False, "message": "No active teleoperation session"}

    try:
        joint_positions = get_joint_positions_from_robot(current_robot, current_robot_model)
        return {"success": True, "joint_positions": joint_positions, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return {"success": False, "message": f"Failed to get joint positions: {str(e)}"}
