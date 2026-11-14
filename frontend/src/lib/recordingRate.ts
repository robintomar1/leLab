/** Frames per second a new dataset is recorded at unless the user picks another. */
export const DEFAULT_RECORDING_FPS = 30;
// The backend's bounds (lelab/record.py).
export const MIN_RECORDING_FPS = 1;
export const MAX_RECORDING_FPS = 240;

export const recordingFpsInRange = (fps: number) =>
  Number.isFinite(fps) && fps >= MIN_RECORDING_FPS && fps <= MAX_RECORDING_FPS;

/**
 * The rate a recorded take actually ran at, as the backend reports it. Frames
 * are stamped at the target rate regardless, so a take that fell short plays
 * back faster than it happened.
 */
export interface LoopRate {
  episode: number;
  target_fps: number;
  achieved_hz: number;
  slow_ticks_percent: number;
  /** Mean cost of one pass of the loop, pacing sleep excluded. */
  mean_work_ms?: number | null;
  kept_up: boolean;
}
