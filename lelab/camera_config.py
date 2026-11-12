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

"""Translate native capture settings into LeRobot camera settings."""

import os
from pathlib import Path
from typing import Any


def validate_camera_rotations(cameras: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reject invalid OpenCV rotations without changing the API payload."""
    for name, config in cameras.items():
        if config.get("type", "opencv") != "opencv":
            continue
        rotation = config.get("rotation", 0)
        if type(rotation) is not int or rotation not in (0, 90, 180, 270):
            raise ValueError(f"Camera '{name}' rotation must be an integer: 0, 90, 180, or 270.")
    return cameras


def lerobot_camera_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return LeRobot output dimensions and its signed rotation value.

    The API dimensions describe the native capture mode. LeRobot camera config
    dimensions describe the rotated output, so quarter-turns swap them.
    """
    validate_camera_rotations({"camera": config})
    settings = config.copy()
    if config.get("type", "opencv") != "opencv":
        return settings

    rotation = config.get("rotation", 0)
    if rotation in (90, 270):
        settings["width"], settings["height"] = config.get("height"), config.get("width")
    # Preserve omission for legacy CLI payloads. LeRobot defaults to no rotation.
    if "rotation" in config:
        settings["rotation"] = -90 if rotation == 270 else rotation
    return settings


def camera_source(name: str, config: dict[str, Any]) -> int | Path:
    """What to open for a camera: its device path when it has one, else its index.

    An index is whatever order the cameras were plugged in, so it moves when
    they are replugged. `camera_path` is a name that follows the camera
    (a udev link such as /dev/v4l/by-path/...). A camera saved with a path
    that is not there is unplugged: falling back to its index would open some
    other camera and record it under this one's name, so that is an error.
    """
    path = config.get("camera_path")
    if not path:
        return config.get("camera_index", 0)
    if not os.path.exists(path):
        raise ValueError(f"Camera '{name}' is not connected: {path} does not exist.")
    return Path(path)
