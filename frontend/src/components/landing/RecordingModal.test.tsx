// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import RecordingModal from "./RecordingModal";
import RecordingRate from "@/components/recording/RecordingRate";
import { RobotRecord } from "@/hooks/useRobots";
import { CameraConfig } from "@/lib/cameraConfig";

vi.mock("@/contexts/HfAuthContext", () => ({ useHfAuth: () => ({ auth: { status: "unauthenticated" } }) }));
vi.mock("@/components/recording/CameraConfiguration", () => ({ default: () => null }));
vi.mock("./CameraBindings", () => ({ default: () => null }));
vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) => open ? <div>{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
}));

const camera: CameraConfig = { id: "head", name: "head", type: "opencv", device_id: "d", camera_index: 2, width: 640, height: 480, fps: 30, rotation: 0 };
const robot: RobotRecord = {
  name: "rebot", robot_model: "rebot_mit" as const, leader_port: "leader", follower_port: "follower",
  leader_config: "leader.json", follower_config: "follower.json", is_clean: true, cameras: [camera],
};
const bindings = { bindings: {}, bind: vi.fn(), allBound: true, availableCameras: [], rotationFor: () => 0, setRotation: vi.fn(), toCameraDict: () => ({}) };

let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div");
  root = createRoot(container);
});
afterEach(() => act(() => root.unmount()));

function render(props: Partial<React.ComponentProps<typeof RecordingModal>> = {}) {
  const setFps = vi.fn();
  act(() => root.render(
    <RecordingModal
      open onOpenChange={vi.fn()} robot={robot} appendTo={null}
      cameraBindings={bindings as never}
      datasetName="d" setDatasetName={vi.fn()} singleTask="t" setSingleTask={vi.fn()}
      numEpisodes={5} setNumEpisodes={vi.fn()} episodeTimeS={60} setEpisodeTimeS={vi.fn()}
      resetTimeS={15} setResetTimeS={vi.fn()} fps={30} setFps={setFps}
      streamingEncoding setStreamingEncoding={vi.fn()} cameras={[camera]} setCameras={vi.fn()}
      onStart={vi.fn()} {...props}
    />,
  ));
  return setFps;
}
const rateField = () => container.querySelector<HTMLInputElement>("#recordingFps")!;
const startButton = () => Array.from(container.querySelectorAll("button")).find((b) => /start/i.test(b.textContent ?? ""))!;

test("a new dataset's rate can be set, and defaults to what it is given", () => {
  const setFps = render();
  expect(rateField().value).toBe("30");
  expect(rateField().disabled).toBe(false);
  const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
  act(() => { setValue.call(rateField(), "60"); rateField().dispatchEvent(new Event("input", { bubbles: true })); });
  expect(setFps).toHaveBeenCalledWith(60);
});

test("a rate above the camera's says the images will repeat", () => {
  render({ fps: 60 });
  expect(container.textContent).toContain("A camera here delivers 30 fps");
  act(() => root.unmount());
  root = createRoot(container);
  render({ fps: 30 });
  expect(container.textContent).not.toContain("A camera here delivers");
});

test.each([0, 241])("a rate of %s is refused before the session starts", (fps) => {
  render({ fps });
  expect(container.querySelector('[role="alert"]')?.textContent).toContain("Enter a rate from 1 to 240");
  expect(startButton().disabled).toBe(true);
});

test("appending shows the dataset's rate and does not let it change", () => {
  render({ fps: 50, appendTo: { repoId: "local/d", numEpisodes: 3, fps: 50, imageFeatures: {} } });
  expect(rateField().value).toBe("50");
  expect(rateField().disabled).toBe(true);
  expect(container.textContent).toContain("Fixed by the dataset");
});

test("the recording page shows the target, then what the last take achieved", () => {
  act(() => root.render(<RecordingRate targetFps={60} />));
  expect(container.textContent).toContain("60 Hz");
  const kept = { episode: 1, target_fps: 60, achieved_hz: 59.2, slow_ticks_percent: 0.3, kept_up: true };
  act(() => root.render(<RecordingRate targetFps={60} loopRate={kept} />));
  expect(container.textContent).toContain("59.2 / 60 Hz");
  expect(container.querySelector(".text-amber-400")).toBeNull();
  act(() => root.render(<RecordingRate targetFps={60} loopRate={{ ...kept, achieved_hz: 41.5, slow_ticks_percent: 88, kept_up: false }} />));
  expect(container.textContent).toContain("41.5 / 60 Hz");
  expect(container.querySelector(".text-amber-400")).not.toBeNull();
  expect(container.querySelector("[title]")?.getAttribute("title")).toContain("plays back faster than it happened");
  act(() => root.render(<RecordingRate targetFps={60} loopRate={{ ...kept, mean_work_ms: 12.5 }} />));
  expect(container.querySelector("[title]")?.getAttribute("title")).toContain("tops out near 80 Hz");
});
