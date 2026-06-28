"""
sim_env_v4.py — DLO-Only MuJoCo Environment (Phase 1v4)
=========================================================
Single scene XML (dlo_scene.xml) with MuJoCo native cable plugin (DER physics).
All 7 scenarios are driven purely by Python via xfrc_applied (external forces
on cable body nodes) — no broken weld constraints, no robot arms.

Public API
----------
env = DLOEnvV2()
env.step(n)                      # advance physics
env.get_cable_keypoints(n_kp)    # (n_kp, 3) world positions along cable
env.apply_force(body_idx, force) # apply (3,) force to cable body i
env.apply_torque(body_idx, torq) # apply (3,) torque to cable body i
env.clear_forces()               # zero all xfrc_applied
env.render_camera(cam)           # (rgb HxWx3, depth HxW)
env.render_camera_rgb_only(cam)  # rgb only (faster)
env.get_camera_intrinsics(cam)   # dict {fx,fy,cx,cy}
env.get_camera_extrinsics(cam)   # dict {R,t}
env.kick(scale)                  # random velocity impulse to all cable joints
env.reset()                      # reset to initial state
env.close()
"""

import os
import numpy as np
import mujoco

_HERE     = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_HERE, "models", "dlo_scene.xml")

IMG_W, IMG_H = 640, 480
CAMERAS = ["overhead_cam", "front_cam", "side_cam"]

SCENARIO_NAMES = {
    "s1_free_hang":   "S1: Free Hang & Sag",
    "s2_forced_bend": "S2: Forced Bending",
    "s3_twist":       "S3: Twisting / Writhe",
    "s4_buckling":    "S4: Euler Buckling",
    "s5_tangling":    "S5: Tangling / Self-Contact",
    "s6_snap":        "S6: Snap-Through",
    "s7_plastic":     "S7: Plastic Deformation",
}
SCENARIO_LIST = list(SCENARIO_NAMES.keys())


class DLOEnvV2:
    def __init__(self, img_w=IMG_W, img_h=IMG_H):
        os.chdir(os.path.join(_HERE, "models"))
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)
        self.img_w = img_w
        self.img_h = img_h
        self._renderer = mujoco.Renderer(self.model, height=img_h, width=img_w)

        # Cache cable body IDs in order (B_first, B_1 .. B_last)
        self._cable_ids = self._find_cable_bodies()
        self.n_cable_bodies = len(self._cable_ids)

        # Cache camera IDs
        self._cam_ids = {}
        for cam in CAMERAS:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            if cid >= 0:
                self._cam_ids[cam] = cid

        # Plastic deformation state: rest qpos (updated in S7)
        self._rest_qpos = self.data.qpos.copy()

        mujoco.mj_forward(self.model, self.data)

    def _find_cable_bodies(self):
        """Return body IDs of cable composite in arc-length order."""
        ids = []
        # B_first is index 0, then B_1..B_N, then B_last
        first_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "B_first")
        if first_id >= 0:
            ids.append(first_id)
        i = 1
        while True:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"B_{i}")
            if bid < 0:
                break
            ids.append(bid)
            i += 1
        last_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "B_last")
        if last_id >= 0:
            ids.append(last_id)
        return ids

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def apply_force(self, body_idx: int, force: np.ndarray):
        """Apply (3,) world-frame force to cable body at index body_idx (0=root, -1=tip)."""
        n = len(self._cable_ids)
        idx = body_idx % n
        self.data.xfrc_applied[self._cable_ids[idx], :3] = force

    def apply_torque(self, body_idx: int, torque: np.ndarray):
        """Apply (3,) world-frame torque to cable body at index body_idx."""
        n = len(self._cable_ids)
        idx = body_idx % n
        self.data.xfrc_applied[self._cable_ids[idx], 3:] = torque

    def clear_forces(self):
        self.data.xfrc_applied[:] = 0.0

    def kick(self, scale=2.0):
        """Random velocity impulse to all cable joints — makes cable swing."""
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            if name.startswith("J_") or "cable" in name.lower():
                dof = self.model.jnt_dofadr[i]
                n_dof = 3 if self.model.joint(i).type == mujoco.mjtJoint.mjJNT_BALL else 1
                self.data.qvel[dof:dof+n_dof] += np.random.uniform(-scale, scale, n_dof)

    def get_cable_keypoints(self, n_kp=20) -> np.ndarray:
        """(n_kp, 3) world positions uniformly sampled along cable."""
        if not self._cable_ids:
            return np.zeros((n_kp, 3))
        pos = np.array([self.data.xpos[bid] for bid in self._cable_ids])
        idx = np.round(np.linspace(0, len(pos)-1, n_kp)).astype(int)
        return pos[idx].copy()

    def render_camera(self, cam_name="overhead_cam"):
        if cam_name not in self._cam_ids:
            return (np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8),
                    np.zeros((self.img_h, self.img_w), dtype=np.float32))
        self._renderer.update_scene(self.data, camera=cam_name)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=cam_name)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return rgb, depth

    def render_camera_rgb_only(self, cam_name="overhead_cam"):
        if cam_name not in self._cam_ids:
            return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self._renderer.update_scene(self.data, camera=cam_name)
        return self._renderer.render().copy()

    def get_camera_intrinsics(self, cam_name):
        if cam_name not in self._cam_ids:
            f = float(self.img_w)
            return dict(fx=f, fy=f, cx=self.img_w/2, cy=self.img_h/2)
        cid = self._cam_ids[cam_name]
        fovy_rad = np.deg2rad(self.model.cam_fovy[cid])
        fy = (self.img_h / 2.0) / np.tan(fovy_rad / 2.0)
        return dict(fx=fy, fy=fy, cx=self.img_w/2.0, cy=self.img_h/2.0)

    def get_camera_extrinsics(self, cam_name):
        if cam_name not in self._cam_ids:
            return dict(R=np.eye(3), t=np.zeros(3))
        cid = self._cam_ids[cam_name]
        R_cw = self.data.cam_xmat[cid].reshape(3, 3).T
        t    = self.data.cam_xpos[cid].copy()
        return dict(R=R_cw, t=t)

    def close(self):
        del self._renderer


# ---------------------------------------------------------------------------
# T1.6 — ScenarioSequencer: auto-cycles all 7 DER scenarios with controllers
# ---------------------------------------------------------------------------

class ScenarioSequencer:
    """
    Cycles the DLOEnvV2 through all 7 DER deformation scenarios automatically.
    Each scenario runs for hold_frames, then transitions to the next.

    Usage:
        seq = ScenarioSequencer(env)
        for frame in range(...):
            seq.step(frame)          # applies forces for current scenario
            name = seq.current_name()
    """

    def __init__(self, env: DLOEnvV2, hold_frames: int = 500):
        self._env   = env
        self._hold  = hold_frames
        self._idx   = 0
        self._local = 0          # frame within current scenario
        self._state = {}         # per-scenario controller state

    def step(self, global_frame: int = 0):
        env   = self._env
        N     = env.n_cable_bodies
        frame = self._local
        sc    = SCENARIO_LIST[self._idx]

        if sc == "s1_free_hang":
            env.clear_forces()
            if frame % 80 == 0:
                env.kick(scale=3.0)

        elif sc == "s2_forced_bend":
            env.clear_forces()
            t = frame * 0.025
            tip_y = 0.5 * np.sin(t)
            tip_z = 0.3 * np.sin(2*t)
            tip_pos = env.data.xpos[env._cable_ids[-1]]
            target  = np.array([tip_pos[0], tip_y, 1.0 + tip_z])
            F_tip   = 8.0*(target-tip_pos) - 0.5*env.data.cvel[env._cable_ids[-1], :3]
            env.apply_force(-1, F_tip)
            mid_pos = env.data.xpos[env._cable_ids[N//2]]
            F_mid   = 3.0*(np.array([mid_pos[0], -0.25*np.sin(t),
                                      1.0+0.15*np.sin(2*t)]) - mid_pos)
            env.apply_force(N//2, F_mid)

        elif sc == "s3_twist":
            env.clear_forces()
            mag = 0.5 + 0.3*np.sin(frame*0.01)
            env.apply_torque(-1, np.array([mag, 0, 0]))
            env.apply_torque(-2, np.array([mag*0.5, 0, 0]))
            tip_pos  = env.data.xpos[env._cable_ids[-1]]
            root_pos = env.data.xpos[env._cable_ids[0]]
            pull = 0.5*(root_pos-tip_pos)/(np.linalg.norm(root_pos-tip_pos)+1e-6)
            env.apply_force(-1, pull)

        elif sc == "s4_buckling":
            env.clear_forces()
            phase    = frame % 300
            compress = min(phase/50., 1.) if phase < 200 else max(0., 1.-(phase-200)/100.)
            tip_pos  = env.data.xpos[env._cable_ids[-1]]
            root_pos = env.data.xpos[env._cable_ids[0]]
            ax = (root_pos-tip_pos); ax /= (np.linalg.norm(ax)+1e-6)
            env.apply_force(-1, 5.0*compress*ax)
            if 5 < phase < 20:
                env.apply_force(N//2, np.array([0, 0.15, 0]))
            if phase > 20:
                for i in range(N//4, 3*N//4):
                    env.apply_force(i, np.array([0, 0.04*compress, 0]))

        elif sc == "s5_tangling":
            env.clear_forces()
            if frame % 30 == 0:
                self._state['forces'] = np.random.uniform(-2.5, 2.5, (N, 3))
                self._state['forces'][:, 0] *= 0.3
            forces = self._state.get('forces', np.zeros((N, 3)))
            for i in range(N):
                env.apply_force(i, forces[i])
            if frame % 60 == 0:
                env.kick(scale=2.0)

        elif sc == "s6_snap":
            env.clear_forces()
            phase = frame % 400
            if phase < 150:
                mag = min(phase/30., 1.)*15.
                env.apply_force(N//2, np.array([0, mag, 0]))
                tip_pos = env.data.xpos[env._cable_ids[-1]]
                env.apply_force(-1, 3.0*(np.array([0.5, 0, 1.0])-tip_pos))
            elif phase < 180:
                env.apply_force(N//2, np.array([0, -8., 0]))
            elif phase < 350:
                tip_pos = env.data.xpos[env._cable_ids[-1]]
                env.apply_force(-1, 2.0*(np.array([0.5, 0, 1.0])-tip_pos))
            else:
                env.apply_force(N//2, np.array([0, -5., 0]))

        elif sc == "s7_plastic":
            env.clear_forces()
            t = frame * 0.02
            bend_pos = int((N*0.2)+(N*0.6)*(0.5+0.5*np.sin(t*0.3)))
            bend_pos = max(1, min(N-2, bend_pos))
            mag = 6.0 + 3.0*np.sin(t)
            env.apply_force(bend_pos,   np.array([0, mag, 0.5*np.sin(t*2)]))
            env.apply_force(bend_pos+1, np.array([0, mag*0.5, 0]))
            if frame % 10 == 0:
                import mujoco as _mj
                for i in range(env.model.njnt):
                    if env.model.joint(i).type == _mj.mjtJoint.mjJNT_BALL:
                        adr = env.model.jnt_qposadr[i]
                        q   = env.data.qpos[adr:adr+4]
                        ang = 2.0*np.arccos(np.clip(abs(q[0]), 0, 1))
                        if ang > 0.2:
                            env.model.qpos0[adr:adr+4] = (q*0.3
                                + env.model.qpos0[adr:adr+4]*0.7)

        # Advance local counter; cycle scenario when hold expires
        self._local += 1
        if self._local >= self._hold:
            self._idx   = (self._idx + 1) % len(SCENARIO_LIST)
            self._local = 0
            self._state = {}
            env.reset()
            env.step(200)   # re-settle after reset

    def current_name(self) -> str:
        return SCENARIO_NAMES[SCENARIO_LIST[self._idx]]

    def current_key(self) -> str:
        return SCENARIO_LIST[self._idx]

    def progress(self) -> float:
        return self._local / max(self._hold, 1)
