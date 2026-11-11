// @vitest-environment jsdom
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { Object3D } from "three";
import { describe, expect, it } from "vitest";
import { ROBOT_MODELS, meshUrl, robotModelInfo } from "./robotModels";

const PUBLIC_DIR = resolve(__dirname, "../../public");

describe("robot model 3D models", () => {
  it("shows the reBot arm for every reBot model and the SO-101 otherwise", () => {
    for (const key of ["rebot_default", "rebot_high_pos_gains", "rebot_mit", "rebot_mit_ff"]) {
      expect(robotModelInfo(key).urdf.path).toBe("/rebot-b601-urdf/urdf/ReBot_Arm_DM.urdf");
    }
    expect(robotModelInfo("so101").urdf.path).toBe("/so-101-urdf/urdf/so101_new_calib.urdf");
    expect(robotModelInfo(undefined).urdf).toBe(ROBOT_MODELS.so101.urdf);
  });

  it("serves a package:// mesh from the model's directory", () => {
    const so101 = ROBOT_MODELS.so101.urdf;
    // What urdf-loader hands the mesh loader for package://so_arm_description/meshes/x.stl
    expect(meshUrl(so101, "//so_arm_description/meshes/x.stl")).toBe("/so-101-urdf/meshes/x.stl");
    expect(meshUrl(so101, "package://so_arm_description/meshes/x.stl")).toBe("/so-101-urdf/meshes/x.stl");
    expect(meshUrl(so101, "/so-101-urdf/meshes/x.stl")).toBe("/so-101-urdf/meshes/x.stl");
  });

  it("leaves a mesh path relative to the URDF alone", () => {
    const url = "/rebot-b601-urdf/urdf/../meshes/visual/colored_link1_matte_black.stl";
    expect(meshUrl(ROBOT_MODELS.rebot_mit.urdf, url)).toBe(url);
  });

  it("ships every mesh each model's URDF references, at the URL the viewer fetches", () => {
    for (const { urdf } of Object.values(ROBOT_MODELS)) {
      const text = readFileSync(join(PUBLIC_DIR, urdf.path), "utf8");
      const meshes = [...text.matchAll(/<mesh\s+filename="([^"]+)"/g)].map((m) => m[1]);
      expect(meshes.length).toBeGreaterThan(0);
      for (const filename of meshes) {
        // As urdf-loader resolves it (package "/"), then as the viewer rewrites it.
        const resolved = filename.startsWith("package://")
          ? `//${filename.slice("package://".length)}`
          : `${dirname(urdf.path)}/${filename}`;
        const served = new URL(meshUrl(urdf, resolved), "http://localhost").pathname;
        expect(existsSync(join(PUBLIC_DIR, served)), served).toBe(true);
      }
    }
  });
});

describe("the reBot URDF in urdf-loader", () => {
  it("parses, with the joints the backend drives", async () => {
    const { default: URDFLoader } = await import("urdf-loader");
    const { path } = ROBOT_MODELS.rebot_mit.urdf;
    const requested: string[] = [];
    const loader = new URDFLoader();
    loader.loadMeshCb = (url, _manager, done) => {
      requested.push(url);
      done(new Object3D());
    };
    loader.workingPath = `${dirname(path)}/`;
    const robot = loader.parse(readFileSync(join(PUBLIC_DIR, path), "utf8"));

    expect(requested).toHaveLength(34);
    for (const url of requested) {
      expect(existsSync(join(PUBLIC_DIR, new URL(url, "http://localhost").pathname)), url).toBe(true);
    }
    for (const name of ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]) {
      expect(robot.joints[name].jointType).toBe("revolute");
    }
    expect(robot.joints.finger_left.jointType).toBe("prismatic");
    expect(robot.joints.finger_right.jointType).toBe("prismatic");

    // Open: the left finger slides towards +y and the right one towards -y.
    robot.setJointValue("finger_left", 0.0285);
    robot.setJointValue("finger_right", -0.0285);
    expect(robot.joints.finger_left.position.y).toBeCloseTo(0.0285);
    expect(robot.joints.finger_right.position.y).toBeCloseTo(-0.0285);
  });
});
