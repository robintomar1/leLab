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

"""Gravity torques of the reBot B601 (Damiao build), for feed-forward in MIT mode.

MIT mode is a spring and damper to the target: nothing in it carries the arm's
weight, so a loaded joint settles where the spring balances gravity, several
degrees off. Adding the torque gravity asks of each joint lets the spring deal
with motion only.

The mass model is the one in Seeed's `ReBot_Arm_DM.urdf` (reBotArm_control_py,
`urdf/DM/urdf/`): the six arm joints, each link's mass and centre of mass, and
the gripper as a rigid 0.5 kg load. Joint angles are the motors' own, which is
how that URDF defines them. The computation is the textbook one, so there is no
dependency on a dynamics library: for joint i, the moment about its axis of the
weight of every link beyond it.
"""

import math

import numpy as np

GRAVITY_M_S2 = 9.81

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")

# Per joint, in order: where it sits in its parent link (xyz in metres, roll-pitch-yaw
# in radians), its axis, and the link it carries (mass in kg, centre of mass in metres).
_CHAIN = (
    (
        (-8.416e-05, 0.0, 0.08465),
        (0.0, 0.0, 0.0),
        (0, 0, 1),
        0.1613,
        (0.000113614552951627, -0.000616319527051323, 0.0236476372671394),
    ),
    (
        (0.020084, 0.031625, 0.05555),
        (-1.5708, 0.0, 0.0),
        (0, 0, -1),
        1.3266,
        (-0.13225622308888, -0.0030617036386309, -0.0308306967030205),
    ),
    (
        (-0.264, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0, 0, 1),
        0.8353,
        (0.121040035791843, -0.0536211076627949, -0.0310137854608077),
    ),
    (
        (0.2426, -0.054, -0.001625),
        (0.0, 0.0, 0.0),
        (0, 0, 1),
        0.5200,
        (0.0608200956293136, -0.0511711906613122, -0.030299458623927),
    ),
    (
        (0.078308, -0.0375, -0.03),
        (-1.5708, 0.0, 0.0),
        (0, 0, 1),
        0.3830,
        (-0.00502802058982517, 1.73866206692364e-06, 0.0386233236326755),
    ),
    (
        (0.028008, 0.0, 0.04),
        (0.0, 1.5708, 0.0),
        (0, 0, 1),
        0.3663,
        (3.76418727127126e-06, -0.000100908819946677, 0.0253308606425965),
    ),
)
# The gripper, fixed to the last link: (xyz, rpy) of its frame, mass, centre of mass.
_END = (
    (0.0, 0.0, 0.15539),
    (0.0, -1.5708, 3.1415),
    0.5,
    (-0.0737654295815033, -9.5080995865868e-06, 7.04327840286845e-06),
)


def _rotation_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _rotation_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


_ORIGINS = tuple(
    (np.array(xyz), _rotation_rpy(*rpy), np.array(axis, dtype=float)) for xyz, rpy, axis, _, _ in _CHAIN
)
_LINKS = tuple((mass, np.array(com)) for *_, mass, com in _CHAIN)
_END_ORIGIN = (np.array(_END[0]), _rotation_rpy(*_END[1]))
_END_LINK = (_END[2], np.array(_END[3]))


def gravity_torques(joint_angles_rad) -> np.ndarray:
    """Torque (N·m) each arm joint must supply to hold the arm still at these angles."""
    rotation, position = np.eye(3), np.zeros(3)
    axes, pivots, loads = (
        [],
        [],
        [],
    )  # per joint: world axis, a point on it; per link: (mass, world centre of mass)
    for (xyz, origin_rotation, axis), angle, (mass, com) in zip(
        _ORIGINS, joint_angles_rad, _LINKS, strict=True
    ):
        position = position + rotation @ xyz
        rotation = rotation @ origin_rotation
        axes.append(rotation @ axis)
        pivots.append(position)
        rotation = rotation @ _rotation_axis(axis, float(angle))
        loads.append((mass, position + rotation @ com))
    end_position = position + rotation @ _END_ORIGIN[0]
    end_mass, end_com = _END_LINK
    # The gripper rides on the last joint: fold it into that joint's load.
    last_mass, last_com = loads[-1]
    end_world = end_position + rotation @ _END_ORIGIN[1] @ end_com
    loads[-1] = (last_mass + end_mass, (last_mass * last_com + end_mass * end_world) / (last_mass + end_mass))

    down = np.array([0.0, 0.0, -GRAVITY_M_S2])
    torques = np.zeros(len(axes))
    for i, (axis, pivot) in enumerate(zip(axes, pivots, strict=True)):
        moment = sum(np.cross(com - pivot, mass * down) for mass, com in loads[i:])
        torques[i] = -float(axis @ moment)  # what the motor supplies opposes what gravity applies
    return torques
