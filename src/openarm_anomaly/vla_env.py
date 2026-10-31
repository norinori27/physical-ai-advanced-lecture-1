"""pi05_droid(DROID 形式)の入出力で OpenArm のシミュを動かす薄い環境。

入力(openpi の droid_policy.py): 外部画像・手首画像(224x224)、関節角 7、グリッパ位置 1(0 開 → 1 閉)、指示文。
出力: 8 次元(関節速度 7 + グリッパ位置 1)。DROID の実機側では、速度 × 0.2 rad を 1 ステップの関節変位とし、
グリッパは 0.5 で 2 値化する(droid の robot_ik_solver.py と openpi の examples/droid/main.py)。
変位を「現在の関節角に足す」のか「前回の目標に足す」のかは、読んだコードからは確認できていない(ここでは現在の関節角)。

retarget=True のときは、行動を仮想の Franka の関節に当て、その手先を OpenArm の IK で追う(franka_retarget.py)。
モデルに渡す関節角も仮想 Franka のもの。ready_tcp は開始時の OpenArm の TCP 位置(DROID の初期姿勢の手先をここに重ねる)。
"""
import mujoco
import numpy as np

from .franka_retarget import FrankaRetarget
from .light_mesh import lighten_visual_meshes
from .scene import build_spec
from .scripted import GRIP_CLOSE, GRIP_OPEN, N_ARM, Kinematics, reset_to_ready

DELTA_SCALE = 0.2   # rad / ステップ(DROID 実機側の max_joint_delta)
CONTROL_HZ = 15
IMG = 224


class DroidStyleEnv:
    def __init__(self, box_half=0.035, box_xy=(0.18, -0.36), obstacles=(), delta_scale=DELTA_SCALE,
                 retarget=False, ready_tcp=None, track_yaw=True):
        spec = build_spec(box_half=box_half, timestep=0.004, box_xy=box_xy, obstacles=obstacles)
        lighten_visual_meshes(spec)
        self.m = spec.compile()
        self.m.light_castshadow[:] = 0
        self.d = mujoco.MjData(self.m)
        self.box_half, self.box_xy = box_half, box_xy
        self.delta_scale = delta_scale
        self.substeps = round(1.0 / CONTROL_HZ / self.m.opt.timestep)   # 17 回 = 0.068 s(約 14.7 Hz)
        self.lo, self.hi = self.m.actuator_ctrlrange[:N_ARM, 0], self.m.actuator_ctrlrange[:N_ARM, 1]
        self.retarget, self.ready_tcp, self.track_yaw = retarget, ready_tcp, track_yaw
        self.kin = Kinematics(self.m)
        self.renderer = mujoco.Renderer(self.m, IMG, IMG)
        self.reset()

    def reset(self):
        reset_to_ready(self.m, self.d, self.box_half, self.box_xy)
        if self.ready_tcp is not None:
            q, err = self.kin.solve_global(np.asarray(self.ready_tcp, float))
            assert err < 5e-3, f"ready_tcp unreachable: err={err:.3f}"
            self.d.qpos[:N_ARM] = q
            self.d.ctrl[:N_ARM] = q
            # 箱のヨーを開始時のハンドに合わせる(reset_to_ready は元の開始姿勢のハンドに合わせているため)
            mujoco.mj_forward(self.m, self.d)
            _, R = self.kin.fk(q)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            jb = self.m.joint("box_free").qposadr[0]
            self.d.qpos[jb + 3:jb + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(self.m, self.d)
        self.d.ctrl[N_ARM] = GRIP_OPEN
        self.q_cmd = self.d.qpos[:N_ARM].copy()
        self.rt = FrankaRetarget(*self.kin.fk(self.q_cmd), self.delta_scale) if self.retarget else None
        return self.obs("")

    def _render(self, cam):
        self.renderer.update_scene(self.d, camera=cam)
        return self.renderer.render().copy()

    def gripper_position(self):
        """指の開き(m)を、DROID の向き(0 開 → 1 閉)に直す。"""
        return float(np.clip((GRIP_OPEN - self.d.qpos[N_ARM]) / (GRIP_OPEN - GRIP_CLOSE), 0, 1))

    def obs(self, prompt):
        return {
            "observation/exterior_image_1_left": self._render("exterior"),
            "observation/wrist_image_left": self._render("wrist"),
            "observation/joint_position": (self.rt.q if self.rt is not None else self.d.qpos[:N_ARM]).astype(np.float64).copy(),
            "observation/gripper_position": np.array([self.gripper_position()]),
            "prompt": prompt,
        }

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        if self.rt is None:
            target = np.clip(self.d.qpos[:N_ARM] + np.clip(a[:N_ARM], -1, 1) * self.delta_scale, self.lo, self.hi)
        else:
            p_des, R_des = self.rt.target(a)
            yaw = np.arctan2(R_des[1, 0], R_des[0, 0]) if self.track_yaw else None
            # 手先は真下向きだけを拘束する(シーンの決まり)。Franka 側の傾きは捨てる。
            self.q_cmd = self.kin.solve(self.q_cmd, p_des, iters=10, yaw=yaw)
            target = self.q_cmd
        self.d.ctrl[:N_ARM] = target
        self.d.ctrl[N_ARM] = GRIP_CLOSE if a[N_ARM] > 0.5 else GRIP_OPEN
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
        if self.rt is not None:
            self.rt.sync(*self.kin.fk(self.d.qpos[:N_ARM]))

    def box_pos(self):
        return self.d.qpos[self.m.joint("box_free").qposadr[0]:][:3].copy()

    def close(self):
        self.renderer.close()
