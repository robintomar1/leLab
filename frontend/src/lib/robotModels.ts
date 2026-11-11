// The leader/follower arm pairs LeLab can drive. Mirrors lelab/robot_models.py:
// the backend validates every `robot_model` it is sent against that registry.

export type RobotModelKey = "so101" | "rebot_b601";

export interface RobotModelInfo {
  key: RobotModelKey;
  label: string;
  /**
   * "range": the user sweeps every joint through its range of motion.
   * "zero": the user poses the arm at its zero position, which the motors store.
   */
  calibration: "range" | "zero";
  /** Whether the 3D viewer has a model of this arm. */
  hasUrdf: boolean;
  /** The robot_type LeRobot stamps on datasets recorded with this follower. */
  datasetRobotType: string;
}

export const DEFAULT_ROBOT_MODEL: RobotModelKey = "so101";

export const ROBOT_MODELS: Record<RobotModelKey, RobotModelInfo> = {
  so101: {
    key: "so101",
    label: "SO-101",
    calibration: "range",
    hasUrdf: true,
    datasetRobotType: "so_follower",
  },
  rebot_b601: {
    key: "rebot_b601",
    label: "reBot B601 (Damiao) + reBot Arm 102 leader",
    calibration: "zero",
    hasUrdf: false,
    datasetRobotType: "rebot_b601_follower",
  },
};

export const robotModelInfo = (key: string | null | undefined): RobotModelInfo =>
  ROBOT_MODELS[key as RobotModelKey] ?? ROBOT_MODELS[DEFAULT_ROBOT_MODEL];
