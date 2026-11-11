// The leader/follower arm pairs LeLab can drive. Mirrors lelab/robot_models.py:
// the backend validates every `robot_model` it is sent against that registry.

export type RobotModelKey =
  | "so101"
  | "rebot_default"
  | "rebot_high_pos_gains"
  | "rebot_mit"
  | "rebot_mit_ff";

/** The 3D model of an arm, served from frontend/public. */
export interface UrdfModel {
  /** URL of the URDF. Relative mesh paths in it resolve against this URL. */
  path: string;
  /** ROS package name -> URL of its directory, for the URDF's `package://` mesh paths. */
  packages?: Record<string, string>;
}

/**
 * The URL to fetch a mesh from, given the one urdf-loader resolved. It leaves a
 * `package://` path under a package directory it has to guess; this puts it
 * under the directory the model declares.
 */
export const meshUrl = (model: UrdfModel, url: string): string => {
  for (const [pkg, base] of Object.entries(model.packages ?? {})) {
    const at = url.indexOf(`${pkg}/`);
    if (at !== -1) return `${base}/${url.slice(at + pkg.length + 1)}`;
  }
  return url;
};

const SO101_URDF: UrdfModel = {
  path: "/so-101-urdf/urdf/so101_new_calib.urdf",
  packages: { so_arm_description: "/so-101-urdf" },
};

// Seeed's ReBot_Arm_DM.urdf; its meshes are relative paths (../meshes/...).
const REBOT_B601_URDF: UrdfModel = {
  path: "/rebot-b601-urdf/urdf/ReBot_Arm_DM.urdf",
};

export interface RobotModelInfo {
  key: RobotModelKey;
  label: string;
  /**
   * "range": the user sweeps every joint through its range of motion.
   * "zero": the user poses the arm at its zero position, which the motors store.
   */
  calibration: "range" | "zero";
  /**
   * The model the 3D viewer shows. The backend sends joint values under this
   * URDF's joint names (`urdf_joints` in lelab/robot_models.py).
   */
  urdf: UrdfModel;
  /** The robot_type LeRobot stamps on datasets recorded with this follower. */
  datasetRobotType: string;
}

const rebot = (key: RobotModelKey, variant: string): RobotModelInfo => ({
  key,
  label: `reBot B601 + Arm 102 leader: ${variant}`,
  calibration: "zero",
  urdf: REBOT_B601_URDF,
  datasetRobotType: "rebot_b601_follower",
});

export const DEFAULT_ROBOT_MODEL: RobotModelKey = "so101";

export const ROBOT_MODELS: Record<RobotModelKey, RobotModelInfo> = {
  so101: {
    key: "so101",
    label: "SO-101",
    calibration: "range",
    urdf: SO101_URDF,
    datasetRobotType: "so_follower",
  },
  // The same arms, driven four ways; see lelab/robot_models.py.
  rebot_default: rebot("rebot_default", "default (factory gains)"),
  rebot_high_pos_gains: rebot("rebot_high_pos_gains", "high position gains"),
  rebot_mit: rebot("rebot_mit", "MIT mode"),
  rebot_mit_ff: rebot("rebot_mit_ff", "MIT mode with velocity feed-forward"),
};

export const robotModelInfo = (key: string | null | undefined): RobotModelInfo =>
  ROBOT_MODELS[key as RobotModelKey] ?? ROBOT_MODELS[DEFAULT_ROBOT_MODEL];
