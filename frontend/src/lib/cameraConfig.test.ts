import { describe, expect, test } from "vitest";
import { CameraConfig, cameraRotation, normalizeCameraConfig, physicalCameraKey, previewDeviceId, rotatedDimensions, rotationForCamera, serializeCameras, syncedWithLive } from "./cameraConfig";
const camera: CameraConfig = { id: "wrist", name: "wrist", device_id: "usb-wrist", camera_index: 1, type: "opencv", width: 640, height: 480, fps: 30 };

test("old profiles default to zero without changing capture dimensions", () => {
  expect(normalizeCameraConfig(camera)).toEqual({ ...camera, rotation: 0 });
});
test.each([0, 90, 180, 270] as const)("rotation %s survives profile and session serialization", (rotation) => {
  const saved = JSON.parse(JSON.stringify(normalizeCameraConfig({ ...camera, rotation })));
  expect(serializeCameras([saved]).wrist).toEqual({ type: "opencv", camera_index: 1, width: 640, height: 480, fps: 30, rotation });
});
test.each([null, "90", -90, 45, 360, true, NaN])("invalid rotation %s is not replaced with zero", (value) => {
  expect(() => cameraRotation(value)).toThrow("Camera rotation must be");
});
test.each([0, 90, 180, 270] as const)("capture/output conversion is reversible at %s degrees", (rotation) => {
  const output = rotatedDimensions(640, 480, rotation);
  expect(output).toEqual(rotation === 90 || rotation === 270 ? { width: 480, height: 640 } : { width: 640, height: 480 });
  expect(rotatedDimensions(output.width, output.height, rotation)).toEqual({ width: 640, height: 480 });
});
test("rotation follows a physical camera across index drift and policy reassignment", () => {
  const saved = [{ ...camera, rotation: 180 as const }, { ...camera, id: "top", name: "top", device_id: "usb-top", camera_index: 2, rotation: 90 as const }];
  const wrist = { deviceId: "usb-wrist", index: 2 }, top = { deviceId: "usb-top", index: 1 };
  expect(rotationForCamera(wrist, saved, {})).toBe(180);
  expect(rotationForCamera(top, saved, {})).toBe(90);
  const overrides = { [physicalCameraKey(wrist)]: 270 as const };
  expect(rotationForCamera(wrist, saved, overrides)).toBe(270);
  expect(rotationForCamera(top, saved, overrides)).toBe(90);
  expect(rotationForCamera({ deviceId: "unsaved", index: 5 }, saved, {})).toBe(0);
});
test("a saved camera previews by its device ID, else by cv2 index in another browser", () => {
  const firefox = [{ index: 0, deviceId: "ff-other" }, { index: 1, deviceId: "usb-wrist" }];
  const safari = [{ index: 0, deviceId: "sf-other" }, { index: 1, deviceId: "sf-wrist" }];
  expect(previewDeviceId(camera, firefox)).toBe("usb-wrist");
  expect(previewDeviceId(camera, safari)).toBe("sf-wrist");
  expect(previewDeviceId(camera, [])).toBe("usb-wrist");
  expect(previewDeviceId({ ...camera, camera_index: 5 }, safari)).toBe("");
});

describe("a camera saved with a device path", () => {
  const head = { ...camera, id: "head", name: "head", device_id: "id-a", camera_index: 2, camera_path: "/dev/head_cam" };
  // After a replug the two cameras came up in the other order: the index and
  // the browser device ID now belong to the other camera.
  const swapped = [
    { index: 2, deviceId: "id-a", path: "/dev/wrist_cam" },
    { index: 4, deviceId: "id-b", path: "/dev/head_cam" },
  ];

  test("is found by its path, not by the index or device ID another camera took", () => {
    expect(previewDeviceId(head, swapped)).toBe("id-b");
    expect(syncedWithLive(head, swapped)).toEqual({ ...head, camera_index: 4, device_id: "id-b" });
  });
  test("has no match while unplugged, rather than another camera's", () => {
    expect(previewDeviceId(head, [swapped[0]])).toBe("");
    expect(syncedWithLive(head, [swapped[0]])).toBe(head);
  });
  test("falls back to device ID and index where the backend reports no paths", () => {
    expect(previewDeviceId(head, [{ index: 2, deviceId: "id-a" }])).toBe("id-a");
  });
  test("sends its path to the session", () => {
    expect(serializeCameras([head]).head).toMatchObject({ camera_index: 2, camera_path: "/dev/head_cam" });
  });
  test("keeps its rotation across the swap", () => {
    expect(rotationForCamera(swapped[1], [{ ...head, rotation: 90 }], {})).toBe(90);
  });
});

test("a camera saved before device paths existed picks up the path of its live match", () => {
  const live = [{ index: 1, deviceId: "usb-wrist", path: "/dev/wrist_cam" }];
  expect(syncedWithLive(camera, live)).toEqual({ ...camera, camera_path: "/dev/wrist_cam" });
  expect(syncedWithLive(camera, [{ index: 1, deviceId: "usb-wrist" }])).toBe(camera);
});
