# reBot B601 (Damiao build) model for the LeLab 3D viewer

The URDF and visual meshes of the Seeed Studio reBot B601 arm, shown on the
teleoperation page when a reBot robot is selected.

## Source

Copied from Seeed's `reBotArm_control_py`, directory `urdf/DM/`, at commit
`d4f3591bc0183de03f3ae8651298db0269aadfdd`:
https://github.com/Seeed-Projects/reBotArm_control_py

That repository carried no license file and no license statement for the URDF
or the meshes when this copy was made.

## Differences from the source

- Only what the viewer renders is here: `meshes/visual/` (30 files) and
  `meshes/shared/` (4 files, the finger carriages and travel stops). The
  `collision/` and `mujoco_collision/` meshes are left out.
- `urdf/ReBot_Arm_DM.urdf` has its `<collision>` elements removed, since they
  point at meshes that are not here, and its leading byte-order mark dropped.
  Everything else (links, joints, inertials, visuals, materials) is unchanged.
- The meshes are plain files, not Git LFS: installing LeLab from git does not
  fetch LFS files.

The upstream README of the model follows, unchanged.

---

# ReBot Arm DM 机械描述包

本目录是可独立复制的 B601-DM 机械描述资源。RS/DM 总体差异及跨引擎复用说明见上一级 [`README.md`](../README.md)。

```text
DM/
├── urdf/
│   └── ReBot_Arm_DM.urdf
└── meshes/
    ├── visual/             # 仅用于 URDF 渲染与配色的分件
    ├── collision/          # URDF 碰撞整件
    ├── shared/             # URDF 渲染与 MuJoCo 碰撞共用
    └── mujoco_collision/   # MuJoCo 夹指分段碰撞网格
```

- `urdf/ReBot_Arm_DM.urdf` 是当前 DM 主模型，包含机械臂与夹爪定义。
- `meshes/visual/` 包含 30 个仅用于 URDF 渲染和配色的分件。
- `meshes/collision/` 包含 10 个 URDF 碰撞整件。
- `meshes/shared/` 包含 4 个由 URDF 渲染和 MuJoCo 碰撞共用的夹指滑块、限位块网格。
- `meshes/mujoco_collision/` 包含 6 个 MuJoCo 使用的左右夹指前、中、后段碰撞网格。
- URDF 使用 `../meshes/...` 相对路径，不依赖 ROS package URI，复制整个 `DM/` 目录即可使用。
- ROS 2 运行时模型仍位于 `reBotArm_ros2_DM/src/rebotarm_bringup/description/`，两处模型更新后应同步核对。

例如，从仓库根目录加载：

```python
from pathlib import Path

urdf_path = Path("Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf")
```
