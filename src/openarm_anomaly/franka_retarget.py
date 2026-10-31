"""DROID(Franka)の関節空間の行動を、OpenArm の手先に付け替える(仮想の Franka)。

pi05_droid の状態と行動は Franka の関節空間なので、OpenArm の関節角をそのまま渡すと範囲外になる。
Cloak(arXiv 2606.22836)のベースライン「π0.5-droid + tip-pose retargeting」と同じ考えで、
  1. 仮想の Franka の関節角 qF に、行動(関節速度 × 0.2 rad)を足す
  2. Franka の順運動学で手先の姿勢を求め、OpenArm の座標に移して OpenArm の IK で追う
  3. 動いた後の OpenArm の手先から、Franka の IK で qF を求め直す(届かない・机に当たった分を反映)
モデルには qF を関節角として渡す。

Franka の運動学: Craig の修正 DH(Gaz et al., RA-L 2019 の表と同じ値)。
DROID のグリッパ(Robotiq 2F-85)の TCP は、フランジから約 0.13〜0.15 m(資料で値がばらつく。未確認)。
"""
import numpy as np
from scipy.optimize import least_squares

# (a_{i-1}, d_i, alpha_{i-1}) 関節 1..7
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
FLANGE_D = 0.107
TCP_D = 0.15          # フランジ → Robotiq の指先中央(概算)
Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# DROID の初期姿勢(droid/robot_env.py の reset_joints)
Q_HOME = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])


def _mdh(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0, 0, 0, 1]])


def franka_fk(q):
    """Franka の基部座標での TCP の位置と回転行列。"""
    T = np.eye(4)
    for (a, d, al), th in zip(DH, q):
        T = T @ _mdh(a, d, al, th)
    T = T @ _mdh(0, FLANGE_D + TCP_D, 0, 0)
    return T[:3, 3].copy(), T[:3, :3].copy()


def franka_ik(q_seed, p_des, R_des, w_rot=0.3, w_reg=0.03):
    """境界付き最小二乗。冗長な 1 自由度は q_seed に近い解を選ぶ。"""
    def res(q):
        p, R = franka_fk(q)
        return np.concatenate([p - p_des, w_rot * (R - R_des).ravel(), w_reg * (q - q_seed)])
    return least_squares(res, np.clip(q_seed, Q_LO + 1e-6, Q_HI - 1e-6), bounds=(Q_LO, Q_HI)).x


class FrankaRetarget:
    """仮想 Franka の状態を持ち、行動を OpenArm の手先目標に変換する。

    座標の対応: 回転なし(どちらも +x が前)の平行移動だけ。DROID の初期姿勢の TCP が、
    OpenArm の初期の TCP 位置に重なるように合わせる。回転は初期姿勢どうしの差 C を固定で持つ
    (R_franka = R_openarm @ C)。
    """

    def __init__(self, p_oa0, R_oa0, delta_scale=0.2):
        self.delta_scale = delta_scale
        self.q = Q_HOME.copy()
        p_f0, R_f0 = franka_fk(self.q)
        self.offset = np.asarray(p_oa0) - p_f0       # Franka 座標 + offset = OpenArm の世界座標
        self.C = R_oa0.T @ R_f0
        self.q_cmd = self.q.copy()

    def target(self, action):
        """行動 → OpenArm の TCP 目標(世界座標の位置、回転行列)。"""
        a = np.clip(np.asarray(action[:7], float), -1, 1)
        self.q_cmd = np.clip(self.q + a * self.delta_scale, Q_LO, Q_HI)
        p, R = franka_fk(self.q_cmd)
        return p + self.offset, R @ self.C.T

    def sync(self, p_oa, R_oa):
        """実際の OpenArm の TCP から仮想 Franka の関節角を求め直す。"""
        self.q = franka_ik(self.q_cmd, np.asarray(p_oa) - self.offset, R_oa @ self.C)
        return self.q
