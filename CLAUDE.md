# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

LeLab is a FastAPI + React web interface wrapping the [LeRobot](https://github.com/huggingface/lerobot) framework for leader/follower arms: the SO-101, and the reBot B601 follower with its reBot Arm 102 leader. It exposes teleoperation, dataset recording, calibration, replay, and training as HTTP/WebSocket endpoints, replacing LeRobot's CLI + keyboard-driven flows.

The frontend (React + Vite) lives in [`frontend/`](frontend/). The bundle in `frontend/dist/` is not committed: [setup.py](setup.py) builds it while building the wheel, with Node from the `nodejs-wheel-binaries` build requirement, and ships it as package data (`frontend.__init__.py` makes setuptools treat it as a package); [`lelab/server.py`](lelab/server.py) mounts it as `StaticFiles` at `/` so a single `lelab` process serves both API and UI on `:8000`. The `frontend/` directory is also force-pushed to the [LeLab HF Space](https://huggingface.co/spaces/lerobot/LeLab) by [`.github/workflows/sync_space.yml`](.github/workflows/sync_space.yml) — that's a separate runtime; the Space builds the [Dockerfile](frontend/Dockerfile) (which runs `npm run build` itself).

## Common commands

Install (editable, requires Python ≥3.10):

```bash
pip install -e .
```

`lerobot` is pinned to a ref on `huggingface/lerobot` (currently `v0.7.0`, see [pyproject.toml](pyproject.toml)), installed from git rather than PyPI because the published wheels don't expose the `lerobot-rollout` script that [lelab/rollout.py](lelab/rollout.py) shells out to. Bump the ref when you want a newer upstream release; expect import-path drift (e.g. `lerobot.configs.dataset`, `lerobot.utils.feature_utils`, `lerobot.common.control_utils`) and CLI flag renames (v0.7.0 replaced `--wandb.*` with `--trackio.*`). [tests/test_lerobot_cli.py](tests/test_lerobot_cli.py) runs the training and rollout argv LeLab builds through the installed LeRobot's parsers, so run the tests right after a bump. Since v0.7.0 the arms' motors go through LeRobot's rustypot-backed `SerialMotorsBus`: use its public methods (`set_operating_mode`, `set_half_turn_homings`, `disconnect`), not Feetech SDK internals.

Run servers (entry point defined in [pyproject.toml](pyproject.toml)):

```bash
lelab          # uvicorn on :8000, serves built frontend at /, opens browser
lelab --dev    # spawns Vite dev (:8080) + uvicorn --reload (:8000), opens browser to :8080
```

Editable installs skip the frontend build: `lelab --dev` serves directly from Vite, and `lelab --rebuild` builds `frontend/dist/` with your local Node.js 22 when you want to try the production bundle. The arm meshes in `frontend/public/` (`so-101-urdf/`, `rebot-b601-urdf/`) are kept out of LFS because installing from git does not fetch LFS files.

Run the Python tests with `pytest` (config in [pyproject.toml](pyproject.toml); install dev deps via `pip install -e ".[test]"`). Tests live in [tests/](tests/) and cover request schemas, pure helpers, and idle/mutex branches of the feature handlers — subprocess/thread happy paths and HF Jobs integration are deliberately not unit-tested. Lint with `ruff check` / `ruff format` (config in [pyproject.toml](pyproject.toml)). There is no Python build step; for end-to-end validation, run `lelab` and exercise endpoints (curl or via the frontend).

## Architecture

### Backend module layout (`lelab/`)

[server.py](lelab/server.py) is a thin FastAPI router. Each feature lives in its own module that owns its global state and exposes `handle_*` functions plus a Pydantic request model:

- [record.py](lelab/record.py): dataset recording around LeRobot's `record_loop`; frontend buttons set the same `events` dict LeRobot's keyboard listener would. The worker is a daemon thread, so the server's shutdown hook calls `stop_recording_and_wait()` to let an active session finalize on Ctrl+C.
- [robot_models.py](lelab/robot_models.py): the registry of arm pairs LeLab drives. It builds each model's LeRobot configs and devices and knows how to bring them up (`connect_arms`); the feature modules below go through it and never name a hardware class.
- [teleoperate.py](lelab/teleoperate.py): leader→follower teleoperation loop over the selected robot model's leader and follower.
- [replay.py](lelab/replay.py): plays a recorded episode's actions on the follower arm, an in-process take on `lerobot-replay` so the dataset browser can stop it.
- [calibrate.py](lelab/calibrate.py) — step-by-step web calibration with a `CalibrationManager` singleton and `_step_complete` threading.Event. A `"range"` model (SO-101) records each joint's range of motion; a `"zero"` model (reBot) waits in the `awaiting_zero` status for the user to pose the arm, then runs the driver's own `calibrate()`.
- [train.py](lelab/train.py) — wraps the LeRobot training CLI as a subprocess (psutil for lifecycle, queue for log streaming).
- [datasets.py](lelab/datasets.py) — dataset listing (local cache + Hub, merged under a `source` field) plus the episode-browsing handlers behind `/dataset-episodes`, `/dataset-episode`, `/dataset-frame`, `/dataset-thumbnails`, `/dataset-motion` and `/dataset-video`.
- [episode_media.py](lelab/episode_media.py) — reads the LeRobot v3.0 on-disk layout (`meta/episodes/*.parquet` + `videos/`) and decodes frames with PyAV. Deliberately does **not** go through `LeRobotDataset`, so browsing a dataset never builds a robot config or imports torch. Several episodes share one mp4, so locating an episode resolves both the file and the `from_/to_timestamp` window inside it — which is why `/dataset-video` is addressed by chunk/file rather than by episode. Call `ensure_episode_index(repo_id)` before opening a local dataset with `LeRobotDataset`: LeRobot writes the index only in `finalize()`, and without it reads the dataset as "not downloaded yet" and fetches it from the Hub, which 404s for a dataset that was never pushed.
- [utils/config.py](lelab/utils/config.py) — shared paths and persistence: calibration JSON, saved ports, saved config selections. **Import shared constants from here, do not hardcode paths in feature modules.**

### State model

Each feature module owns module-level globals (e.g. `recording_active`, `teleoperation_active`, `current_robot`) protected by threads/locks where needed. There's no shared session object. Recording, teleoperation, inference and replay all drive the follower, so each start handler checks the others' flags and refuses to run while one is active. Calibration is not part of that check.

### WebSocket broadcast

[server.py](lelab/server.py#L104-L206) defines a single `ConnectionManager` with a background `_broadcast_worker` thread that drains a `queue.Queue` and forwards joint data to all `/ws/joint-data` clients via a thread-local asyncio loop. Feature modules get the manager passed in (e.g. `handle_start_teleoperation(request, manager)`) and call `manager.broadcast_joint_data_sync(data)` from their worker threads. Don't `await` from these threads — use the sync queue method.

### Persistent state on disk

All under `~/.cache/huggingface/lerobot/` (managed in [utils/config.py](lelab/utils/config.py)):

- `calibration/teleoperators/<leader name>/*.json` — leader calibration files (also called "teleop"): `so_leader`, `rebot_102_leader`
- `calibration/robots/<follower name>/*.json` — follower calibration files (also called "robot"): `so_follower`, `rebot_b601_follower`
- `robots/<name>.json` — robot records: `robot_model`, ports, linked calibration files, cameras. Changing a record's model unlinks its calibration files. A camera carries `camera_index` and, on Linux, `camera_path`: a udev link that follows the camera across replugs (`/available-cameras` reports one per camera). Where a path is saved it decides which camera is opened and previewed ([lelab/camera_config.py](lelab/camera_config.py) `camera_source`, [frontend/src/lib/cameraConfig.ts](frontend/src/lib/cameraConfig.ts) `liveCameraFor`); a saved path that is missing is an error, not a fallback to the index.
- `ports/{leader,follower}_port.txt` — last-used serial ports
- `saved_configs/{leader,follower}_config.txt` — last-selected config name

`device_type` in API requests is `"teleop"` or `"robot"` (mapped to leader/follower paths). `robot_type` in port endpoints is `"leader"` or `"follower"`. `robot_model` is the key of a robot model (`"so101"`, `"rebot_high_pos_gains"`, ...). Don't conflate the three vocabularies.

### Calibration files: dual-location pattern

[setup_calibration_files](lelab/utils/config.py#L30-L74) copies user-selected configs from `LEADER_CONFIG_PATH` / `FOLLOWER_CONFIG_PATH` into LeRobot's expected locations under `~/.cache/huggingface/lerobot/calibration/`. Recording, teleoperation, and replay all call this before starting. New features that drive a robot must do the same.

## Hardware target

The supported arm pairs are the entries of `ROBOT_MODELS` in [lelab/robot_models.py](lelab/robot_models.py), mirrored for the UI in [frontend/src/lib/robotModels.ts](frontend/src/lib/robotModels.ts):

- `so101` (default): SO-101 leader and follower, Feetech motors.
- `rebot_default`, `rebot_high_pos_gains`, `rebot_mit`, `rebot_mit_ff`: the Seeed Studio reBot B601 follower (Damiao motors over a serial CAN bridge) with the reBot Arm 102 leader, driven four ways. They are one `family` (`rebot_b601`): same drivers, calibration files and dataset `robot_type`, so switching a record between them keeps its calibration. All need the `rebot` extra (`pip install -e ".[rebot]"`), which pulls `motorbridge`. The 3D viewer shows Seeed's `ReBot_Arm_DM.urdf` ([frontend/public/rebot-b601-urdf/](frontend/public/rebot-b601-urdf/), visual meshes only). Each model's `urdf_joints` in [lelab/robot_models.py](lelab/robot_models.py) says which URDF joints a motor sets and in what unit: the six arm motors are `joint1`..`joint6` in radians, and the gripper motor (0° closed to -270° open) slides `finger_left` and `finger_right` up to 28.5 mm apart, in metres. [frontend/src/lib/robotModels.ts](frontend/src/lib/robotModels.ts) names the URDF the viewer loads for each model.
  - `rebot_default`: position-velocity mode with the motors' factory position-loop gain (`KP_APR` 54). The follower trails the leader by 0.2 s (wrist) to 0.7 s (shoulder, elbow).
  - `rebot_high_pos_gains`: position-velocity mode with `KP_APR` raised as far as the whole arm stays stable (180–270 on shoulder and elbow, 100 on the wrist): roughly half the lag. Don't raise these from single-joint tests: the limit is a whole-arm resonance, measured in [lelab/rebot_tuning.py](lelab/rebot_tuning.py).
  - `rebot_mit`: MIT mode (a spring and damper to the target) with LeRobot's gains, through LeLab's own follower ([lelab/rebot_ff_follower.py](lelab/rebot_ff_follower.py), type `lelab_rebot_b601_ff_follower`). It adds the torque that carries the arm's weight, from the Damiao URDF's mass model in [lelab/rebot_gravity.py](lelab/rebot_gravity.py); without it the loaded joints sag several degrees. `max_relative_target` keeps the target within 20° of the arm, since MIT mode has no speed limit of its own, and `max_dynamic_torque` bounds what the spring and damper together may ask of each joint: unbounded, a violent move asks every big joint for its peak torque at once, the 350 W supply's overload protection cuts out, and the motors restart disabled. The follower trails by about kd/kp seconds.
  - `rebot_mit_ff`: the same, also sending the target's velocity so the damper stops holding the arm back. The rollout subprocess is started through [lelab/rollout_entry.py](lelab/rollout_entry.py), which registers the follower type before LeRobot parses `--robot.type`.

  All four use LeLab's follower, in every control mode, because it reads each motor's status with its feedback: a Damiao motor that trips a protection (over/under-voltage, over-current, over-temperature, overload) stops producing torque and LeRobot's driver never notices. The follower raises `MotorFaultError` (lelab/utils/devices.py) naming the joint and the protection; teleoperation lets it through instead of treating it as a display error, and `/teleoperation-status` reports it as `last_error`. The server also logs to `~/.cache/huggingface/lerobot/lelab_logs/lelab.log`.

  Tight tracking is what teleoperation wants and not obviously what a policy wants: "Tune to Learn" (arXiv:2604.02523) finds imitation policies do best on compliant, well-damped controllers, where action errors turn into small state errors. Train and deploy with the model the data was recorded with: the recorded actions only mean what they did under that controller.

  Position-velocity mode closes the gap to a target at a bounded speed, which suits sessions started from a button. Before anything connects to the follower, `prepare_follower` writes the model's `position_kp` to the motors (RAM only) through [lelab/rebot_tuning.py](lelab/rebot_tuning.py). `rebot_default` writes the factory value too, because a raised gain stays in the motors until they are power-cycled. It opens the port itself, so it has to run while the port is free — including before the rollout subprocess.

Every request that drives an arm carries `robot_model` (default `so101`), taken from the robot record. To add another way of driving existing arms, add an entry to both registries. For new hardware, also give `make_follower_config` / `make_leader_config` a branch for its `family` and pick the `connect_arms` path that fits its calibration style.

## Phone camera / HTTPS

Phone-as-camera streaming requires HTTPS. See [PHONE_CAMERA_SETUP.md](PHONE_CAMERA_SETUP.md) and [HTTPS_SETUP.md](HTTPS_SETUP.md). Self-signed certs go in `certs/`. The default `lelab-fullstack` does **not** start with HTTPS — for phone cameras run uvicorn manually with `--ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem`.
