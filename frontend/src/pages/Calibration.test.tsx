// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import Calibration from "./Calibration";

const mocks = vi.hoisted(() => ({ fetcher: vi.fn(), toast: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetcher }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: mocks.toast }) }));
vi.mock("react-router-dom", () => ({
  useNavigate: () => vi.fn(),
  useLocation: () => ({ state: { robot_name: "arm" } }),
}));
// Native selects, so a choice is a change event rather than Radix popup plumbing.
vi.mock("@/components/ui/select", () => ({
  Select: ({ value, onValueChange, disabled, children }: { value: string; onValueChange: (value: string) => void; disabled?: boolean; children: React.ReactNode }) => (
    <select value={value ?? ""} disabled={disabled} onChange={(event) => onValueChange(event.target.value)}><option value="" />{children}</select>
  ),
  SelectContent: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  SelectItem: ({ value, children }: { value: string; children: React.ReactNode }) => <option value={value}>{children}</option>,
  SelectTrigger: () => null,
  SelectValue: () => null,
}));
vi.mock("@/components/ui/PortDetectionModal", () => ({ default: () => null }));
vi.mock("@/components/recording/CameraConfiguration", () => ({ default: () => null }));
vi.mock("@/components/Logo", () => ({ default: () => null }));

let record: Record<string, unknown>;
let status: Record<string, unknown>;
let root: Root;
let container: HTMLDivElement;

const idle = { calibration_active: false, status: "idle", device_type: null, error: null, message: "", step: 0, total_steps: 1, current_positions: null, recorded_ranges: null };
const json = (body: unknown) => new Response(JSON.stringify(body));

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div"); root = createRoot(container);
  record = { name: "arm", robot_model: "rebot_b601", leader_port: "/dev/ttyUSB0", follower_port: "/dev/ttyACM0", leader_config: "leader.json", follower_config: "", cameras: [], is_clean: false };
  status = idle;
  mocks.fetcher.mockReset(); mocks.toast.mockReset();
  mocks.fetcher.mockImplementation(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/robots/arm")) {
      if (init?.method === "POST") {
        const patch = JSON.parse(init.body as string);
        // The backend unlinks calibration files when the model changes.
        record = { ...record, ...patch, ...(patch.robot_model ? { leader_config: "", follower_config: "" } : {}) };
      }
      return json({ status: "success", robot: record });
    }
    if (url.includes("/calibration-configs/")) return json({ success: true, configs: [{ filename: "follower1.json" }] });
    if (url.endsWith("/start-calibration")) {
      status = { ...idle, calibration_active: true, status: "awaiting_zero", device_type: "robot", step: 1 };
      return json({ success: true });
    }
    if (url.endsWith("/complete-calibration-step")) {
      status = { ...idle, status: "completed" };
      return json({ success: true, message: "Zero position set" });
    }
    if (url.endsWith("/calibration-status")) return json(status);
    return json({});
  });
});
afterEach(() => { act(() => root.unmount()); });

const flush = () => act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
const button = (label: string) => Array.from(container.querySelectorAll("button")).find((b) => b.textContent?.includes(label));
const calls = (suffix: string) => mocks.fetcher.mock.calls.filter(([url]) => (url as string).includes(suffix));
const modelSelect = () => Array.from(container.querySelectorAll("select")).find((s) => Array.from(s.options).some((o) => o.value === "rebot_b601"))!;

async function render() {
  await act(async () => root.render(<Calibration />));
  await flush();
}

test("a reBot arm is calibrated by setting its zero position", async () => {
  await render();

  expect(modelSelect().value).toBe("rebot_b601");
  expect(container.textContent).toContain("Zero-position calibration");
  expect(container.querySelector("video")).toBeNull();
  // The follower is the side still missing a calibration, and its files are listed per model.
  expect(calls("/calibration-configs/").at(-1)![0]).toBe("http://test/calibration-configs/robot?robot_model=rebot_b601");

  await act(async () => button("Start Calibration")!.click());
  await flush();
  const request = JSON.parse(calls("/start-calibration")[0][1].body);
  expect(request).toEqual({ device_type: "robot", port: "/dev/ttyACM0", config_file: "arm", robot_name: "arm", robot_model: "rebot_b601" });

  expect(container.textContent).toContain("Torque is off");
  await act(async () => button("Set Zero Position")!.click());
  expect(calls("/complete-calibration-step")).toHaveLength(1);
});

test("changing the robot model saves it and switches the calibration flow", async () => {
  await render();

  await act(async () => {
    modelSelect().value = "so101";
    modelSelect().dispatchEvent(new Event("change", { bubbles: true }));
  });
  await flush();

  const post = calls("/robots/arm").find(([, init]) => init?.method === "POST")!;
  expect(JSON.parse(post[1].body)).toEqual({ robot_model: "so101" });
  expect(modelSelect().value).toBe("so101");
  expect(container.querySelector("video")).not.toBeNull();
  expect(container.textContent).not.toContain("Zero-position calibration");
  expect(calls("/calibration-configs/").at(-1)![0]).toContain("robot_model=so101");
});
