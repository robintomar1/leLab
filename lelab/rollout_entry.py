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

"""`lerobot-rollout`, with LeLab's own robot types available to `--robot.type`.

LeRobot's CLIs only know the robots LeRobot ships (and installed `lerobot_robot_*`
plugins). Importing LeLab's follower here registers it before the arguments are
parsed, so inference drives the arm exactly as teleoperation and recording did.
"""

from lerobot.scripts.lerobot_rollout import main

from . import rebot_ff_follower  # noqa: F401  (registers its config type)

if __name__ == "__main__":
    main()
