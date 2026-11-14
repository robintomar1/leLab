import React from "react";
import { LoopRate } from "@/lib/recordingRate";

interface RecordingRateProps {
  targetFps: number;
  loopRate?: LoopRate;
}

/**
 * The dataset's rate and, once a take has finished, the rate that take really
 * ran at. Amber when the loop could not hold the rate.
 */
const RecordingRate: React.FC<RecordingRateProps> = ({ targetFps, loopRate }) => {
  // One pass of the loop takes this long, so no rate above 1000 / this can be held.
  const work =
    loopRate?.mean_work_ms != null
      ? ` One pass of the loop took ${loopRate.mean_work_ms} ms on average, so this setup tops out near ` +
        `${Math.floor(1000 / loopRate.mean_work_ms)} Hz.`
      : "";
  const title = !loopRate
    ? `Recording at ${targetFps} frames per second.`
    : loopRate.kept_up
    ? `Episode ${loopRate.episode} ran at ${loopRate.achieved_hz} Hz; the target is ${targetFps}.${work}`
    : `Episode ${loopRate.episode} ran at ${loopRate.achieved_hz} Hz, short of the ${targetFps} Hz target ` +
      `(${loopRate.slow_ticks_percent}% of ticks overran). Frames are still stamped ${targetFps} per second, ` +
      `so it plays back faster than it happened. Record at a lower rate.${work}`;
  return (
    <div className="flex items-baseline gap-2" title={title} aria-label={title}>
      <span className="text-xs font-bold tracking-widest text-gray-500">RATE</span>
      <span
        className={`font-mono text-xl ${
          loopRate && !loopRate.kept_up ? "text-amber-400" : "text-gray-300"
        }`}
      >
        {loopRate ? `${loopRate.achieved_hz} / ${targetFps} Hz` : `${targetFps} Hz`}
      </span>
    </div>
  );
};

export default RecordingRate;
