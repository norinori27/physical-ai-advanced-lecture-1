"""差分 IK によるスクリプト制御のピック&運搬(CPU MuJoCo)。

RL 方策が収束しない場合の保険と、保持の成立確認に使う。
IK は「TCP 位置 + ハンドが真下を向く」の 5 自由度で解き、ヨーは自由にする
(手首の可動域が狭く、ヨーまで固定すると届く範囲がほぼ無くなるため)。
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares

from .scene import STATION_A, STATION_B

N_ARM = 7
GRIP_OPEN, GRIP_CLOSE = 0.044, -0.005
DOWN = np.array([0.0, 0.0, -1.0])
HOVER = 0.15


def waypoints(box_half, hover=HOVER, speed=1.0):
    """(所要時間 s, tcp 目標位置, グリッパ指令) の列。"""
    ax, ay = STATION_A
    bx, by = STATION_B
    z_grasp = box_half
    z_hover = box_half + hover
    s = 1.0 / speed
    return [
        (1.0 * s, (ax, ay, z_hover), GRIP_OPEN),   # 接近
        (1.0 * s, (ax, ay, z_grasp), GRIP_OPEN),   # 降下
        (0.6, (ax, ay, z_grasp), GRIP_CLOSE),       # 把持
        (1.0 * s, (ax, ay, z_hover), GRIP_CLOSE),   # 持ち上げ
        (1.8 * s, (bx, by, z_hover), GRIP_CLOSE),   # 運搬
        (1.0 * s, (bx, by, z_grasp + 0.003), GRIP_CLOSE),  # 置く
        (0.5, (bx, by, z_grasp + 0.003), GRIP_OPEN),  # 解放
        (1.0 * s, (bx, by, z_hover), GRIP_OPEN),    # 退避
    ]


class Kinematics:
    """指令値側の関節角に対して運動学だけで IK を解く(ダイナミクスと切り離す)。"""

    def __init__(self, m):
        self.m = m
        self.d = mujoco.MjData(m)
        self.site = m.site("tcp").id
        self.jac_p = np.zeros((3, m.nv))
        self.jac_r = np.zeros((3, m.nv))
        self.lo = m.jnt_range[:N_ARM, 0]
        self.hi = m.jnt_range[:N_ARM, 1]

    def fk(self, q):
        self.d.qpos[:N_ARM] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, q, p_des, iters=3, max_step=0.1):
        """直前の解から追従する増分 IK(連続な軌道用)。"""
        q = q.copy()
        for _ in range(iters):
            p, R = self.fk(q)
            err = np.concatenate([p_des - p, np.cross(R[:, 2], DOWN)])
            mujoco.mj_jacSite(self.m, self.d, self.jac_p, self.jac_r, self.site)
            J = np.vstack([self.jac_p, self.jac_r])[:, :N_ARM]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -max_step, max_step), self.lo, self.hi)
        return q

    def solve_global(self, p_des, n_starts=20, seed=0):
        """マルチスタートの境界付き最小二乗(初期姿勢を求める用)。"""
        rng = np.random.default_rng(seed)

        def res(q):
            p, R = self.fk(q)
            return np.concatenate([p - p_des, 0.3 * (R[:, 2] - DOWN)])

        best = None
        for _ in range(n_starts):
            r = least_squares(res, rng.uniform(self.lo, self.hi), bounds=(self.lo, self.hi))
            if best is None or r.cost < best.cost:
                best = r
        return best.x, np.linalg.norm(best.fun)


def reset_to_ready(m, d, box_half):
    """ステーション A の上空で下向きの姿勢にし、箱のヨーをその時のハンドに合わせる。"""
    kin = Kinematics(m)
    q, err = kin.solve_global(np.array([*STATION_A, box_half + HOVER]))
    assert err < 5e-3, f"ready pose unreachable: err={err:.3f}"
    # 真下へ降ろした把持姿勢でのハンドのヨー
    q_g = kin.solve(q, np.array([*STATION_A, box_half]), iters=100)
    _, R = kin.fk(q_g)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[:N_ARM] = q
    d.ctrl[:N_ARM] = q
    jb = m.joint("box_free").qposadr[0]
    d.qpos[jb + 3:jb + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(m, d)


class ScriptedController:
    def __init__(self, m, box_half, speed=1.0, ctrl_dt=0.02):
        self.m = m
        self.ctrl_dt = ctrl_dt
        self.kin = Kinematics(m)
        self.wps = waypoints(box_half, speed=speed)
        self.t_ends = np.cumsum([w[0] for w in self.wps])
        self.duration = self.t_ends[-1]
        self.q_cmd = None
        self.start = None

    def target(self, t):
        i = min(int(np.searchsorted(self.t_ends, t)), len(self.wps) - 1)
        t0 = self.t_ends[i - 1] if i > 0 else 0.0
        a = np.clip((t - t0) / self.wps[i][0], 0, 1)
        a = a * a * (3 - 2 * a)  # smoothstep
        p1 = np.array(self.wps[i][1])
        p0 = np.array(self.wps[i - 1][1]) if i > 0 else self.start
        return p0 + a * (p1 - p0), self.wps[i][2], i

    def __call__(self, d, t):
        if self.q_cmd is None:
            self.q_cmd = d.qpos[:N_ARM].copy()
            self.start = self.kin.fk(self.q_cmd)[0]
        p_des, grip, phase = self.target(t)
        self.q_cmd = self.kin.solve(self.q_cmd, p_des)
        d.ctrl[:N_ARM] = self.q_cmd
        d.ctrl[N_ARM] = grip
        return phase


def run_episode(m, d, ctrl, render=None, every=2, camera="overview"):
    """1 サイクル実行し、ログ(関節角・速度・指令・トルク・グリッパ・物体位置)を返す。"""
    n_sub = int(round(ctrl.ctrl_dt / m.opt.timestep))
    steps = int(ctrl.duration / ctrl.ctrl_dt) + 10
    log = {k: [] for k in ["qpos", "qvel", "ctrl", "torque", "grip", "box_pos", "tcp", "tcp_des", "phase"]}
    frames = []
    box = m.body("box").id
    site = m.site("tcp").id
    for k in range(steps):
        phase = ctrl(d, k * ctrl.ctrl_dt)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        log["qpos"].append(d.qpos[:N_ARM].copy())
        log["qvel"].append(d.qvel[:N_ARM].copy())
        log["ctrl"].append(d.ctrl.copy())
        log["torque"].append(d.actuator_force[:N_ARM].copy())
        log["grip"].append(d.qpos[N_ARM])
        log["box_pos"].append(d.xpos[box].copy())
        log["tcp"].append(d.site_xpos[site].copy())
        log["tcp_des"].append(ctrl.target(k * ctrl.ctrl_dt)[0])
        log["phase"].append(phase)
        if render is not None and k % every == 0:
            render.update_scene(d, camera=camera)
            frames.append(render.render())
    return {k: np.array(v) for k, v in log.items()}, frames
