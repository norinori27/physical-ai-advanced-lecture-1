"""「箱を指定した色のマーカーに置く」タスクの変数(位置・ゴール・画像・言語)の定義、サンプル、シーンへの反映。

変数と水準(学習 / 評価だけで使う未知の水準):
  位置   箱・マーカー・開始時の手先の xy。学習は内側の範囲、未知は外側の帯(POS_OUTER)。
  ゴール マーカーの色。学習は blue / green、未知は yellow。
  画像   照明(明るさ・向き)と箱の色。照明: 学習は LIGHT_TRAIN の範囲、未知は暗い・斜め。箱: 学習は orange / red、未知は purple。
  言語   指示の言い回し。学習は TEMPLATES_TRAIN の 3 通り、未知は TEMPLATES_HELDOUT。
評価の条件は CONDITIONS(基準 = 学習の分布の未知の seed、ほかは 1 つの変数だけを未知の水準に外す)。
"""
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

MARKER_RGB = {"blue": (0.2, 0.4, 0.9), "green": (0.2, 0.7, 0.3), "yellow": (0.95, 0.85, 0.15)}
BOX_RGB = {"orange": (0.85, 0.55, 0.2), "red": (0.8, 0.15, 0.15), "purple": (0.55, 0.25, 0.75)}
MARKERS_TRAIN, MARKERS_HELDOUT = ("blue", "green"), ("yellow",)
BOXES_TRAIN, BOXES_HELDOUT = ("orange", "red"), ("purple",)
TEMPLATES_TRAIN = ("put the {box} cube on the {marker} marker",
                   "place the {box} block onto the {marker} square",
                   "move the {box} cube to the {marker} marker")
TEMPLATES_HELDOUT = ("pick up the {box} cube and drop it on the {marker} target",)

MARKER_HALF = 0.045          # マーカーは 9 cm 四方(箱は 7 cm)
N_MARKERS = 2                # 1 シーンに置くマーカーの数(ゴール 1 + 紛らわしい 1)
MIN_SEP = 0.14               # 箱・マーカーの中心どうしの最小距離
# 位置の範囲(真下向きで z 0.035 と 0.20 の両方に届く範囲 x 0.14〜0.38, |y|≦0.36 の内側)
POS_INNER = dict(x=(0.18, 0.32), y=(-0.26, 0.26))
BOX_X_MIN = 0.25             # 箱は台座から離す(x < 0.25 では教師の把持の失敗が 51%)
POS_OUTER = dict(x=(0.14, 0.36), y=(-0.34, 0.34))   # 外側の帯 = OUTER のうち INNER の外
READY_TRAIN = dict(x=(0.16, 0.24), y=(-0.08, 0.08), z=(0.24, 0.28))
LIGHT_TRAIN = dict(diffuse=(0.6, 1.0), xy=0.3)       # 真上 2 m の光源を ±0.3 m ずらす
LIGHT_HELDOUT = (dict(diffuse=0.35, pos=(0.0, 0.0, 2.0)), dict(diffuse=0.8, pos=(1.2, -1.2, 1.0)))


@dataclass
class TaskSpec:
    box_xy: tuple
    box_color: str
    markers: list                # [(色, (x, y)), ...]
    target: str                  # ゴールのマーカーの色
    template: str
    ready_tcp: tuple
    light_diffuse: float = 0.8
    light_pos: tuple = (0.0, 0.0, 2.0)
    box_yaw_noise_deg: float = 0.0
    condition: str = "train"
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def prompt(self):
        return self.template.format(box=self.box_color, marker=self.target)

    @property
    def target_xy(self):
        return dict(self.markers)[self.target]

    def to_dict(self):
        d = asdict(self)
        d["prompt"], d["target_xy"] = self.prompt, list(self.target_xy)
        return d


def _sample_xy(rng, region, n, outer_only=False, inner=POS_INNER, tries=1000):
    """region 内で、互いに MIN_SEP 以上離れた n 点。outer_only なら少なくとも 1 点(先頭)を inner の外にする。"""
    for _ in range(tries):
        pts = [(rng.uniform(*region["x"]), rng.uniform(*region["y"])) for _ in range(n)]
        if pts[0][0] < BOX_X_MIN and not outer_only:   # 先頭(箱)だけ x の下限を上げる
            continue
        if outer_only:
            x, y = pts[0]
            if inner["x"][0] <= x <= inner["x"][1] and inner["y"][0] <= y <= inner["y"][1]:
                continue
        if all(np.hypot(a[0] - b[0], a[1] - b[1]) >= MIN_SEP for i, a in enumerate(pts) for b in pts[i + 1:]):
            return pts
    raise RuntimeError("配置が見つからない")


def sample_task(rng, condition="train", seed=0):
    """condition: train / base / pos_outer / goal_color / light / box_color / language"""
    markers_pool, boxes, templates = MARKERS_TRAIN, BOXES_TRAIN, TEMPLATES_TRAIN
    region, outer = POS_INNER, False
    if condition == "pos_outer":
        region, outer = POS_OUTER, True
    pts = _sample_xy(rng, region, 1 + N_MARKERS, outer_only=outer)
    if condition == "pos_outer":          # 外側に置くのは、箱かゴールのマーカーのどちらか
        pts = pts if rng.random() < 0.5 else [pts[1], pts[0], *pts[2:]]
    colors = [str(c) for c in rng.permutation(markers_pool)][:N_MARKERS]
    if condition == "goal_color":         # ゴールを未知の色にし、もう 1 つは既知の色
        colors = [MARKERS_HELDOUT[0], str(rng.choice(markers_pool))]
    target = colors[0] if condition == "goal_color" else str(rng.choice(colors))
    box_color = BOXES_HELDOUT[0] if condition == "box_color" else str(rng.choice(boxes))
    template = TEMPLATES_HELDOUT[0] if condition == "language" else str(rng.choice(templates))
    ready = tuple(rng.uniform(*READY_TRAIN[k]) for k in "xyz")
    if condition == "light":
        lt = LIGHT_HELDOUT[int(rng.integers(len(LIGHT_HELDOUT)))]
        diffuse, lpos = lt["diffuse"], lt["pos"]
    else:
        diffuse = rng.uniform(*LIGHT_TRAIN["diffuse"])
        lpos = (rng.uniform(-1, 1) * LIGHT_TRAIN["xy"], rng.uniform(-1, 1) * LIGHT_TRAIN["xy"], 2.0)
    return TaskSpec(box_xy=tuple(map(float, pts[0])), box_color=box_color,
                    markers=[(c, tuple(map(float, p))) for c, p in zip(colors, pts[1:])],
                    target=target, template=template, ready_tcp=tuple(map(float, ready)),
                    light_diffuse=float(diffuse), light_pos=tuple(map(float, lpos)),
                    box_yaw_noise_deg=float(rng.uniform(-7, 7)), condition=condition, seed=int(seed))


CONDITIONS = ("base", "pos_outer", "goal_color", "light", "box_color", "language")


def add_markers(spec):
    """build_spec の MjSpec にマーカーのサイトを足し、元のステーションの印は消す。"""
    for s in spec.sites:
        if s.name.startswith("station_"):
            s.rgba = [0, 0, 0, 0]
    for i in range(len(MARKER_RGB)):
        spec.worldbody.add_site(name=f"marker{i}", pos=[0, 0, 0.0005], size=[MARKER_HALF, MARKER_HALF, 0.0005],
                                type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[0, 0, 0, 0])


def apply_task(env, task):
    """マーカー・色・照明を反映し、箱のヨーを「その位置で把持するときのハンドの向き ± ノイズ」に合わせる。"""
    m, d = env.m, env.d
    for i in range(len(MARKER_RGB)):
        m.site_rgba[m.site(f"marker{i}").id] = 0
    for i, (c, (x, y)) in enumerate(task.markers):
        sid = m.site(f"marker{i}").id
        m.site_pos[sid] = [x, y, 0.0005]
        m.site_rgba[sid] = [*MARKER_RGB[c], 1]
    m.geom_rgba[m.geom("box_geom").id] = [*BOX_RGB[task.box_color], 1]
    m.light_diffuse[0] = task.light_diffuse
    m.light_pos[0] = task.light_pos
    direction = -np.asarray(task.light_pos) + [0.25, 0, 0]
    m.light_dir[0] = direction / np.linalg.norm(direction)
    # 箱のヨー: 開始姿勢から箱の真上へ降ろしたときのハンドの向き(ヨー自由の IK)に合わせる
    q = d.qpos[:7].copy()
    q_g = env.kin.solve(q, np.array([*task.box_xy, env.box_half + 0.15]), iters=100)
    q_g = env.kin.solve(q_g, np.array([*task.box_xy, env.box_half]), iters=100)
    _, R = env.kin.fk(q_g)
    yaw = np.arctan2(R[1, 0], R[0, 0]) + np.deg2rad(task.box_yaw_noise_deg)
    jb = m.joint("box_free").qposadr[0]
    d.qpos[jb:jb + 3] = [*task.box_xy, env.box_half]
    d.qpos[jb + 3:jb + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    return float(yaw)


def place_success(env, task, tol=MARKER_HALF):
    """箱がゴールのマーカーの上にあり、机に置かれていて(持ち上げていない)、グリッパが開いている。"""
    box = env.box_pos()
    tx, ty = task.target_xy
    on_target = abs(box[0] - tx) < tol and abs(box[1] - ty) < tol
    resting = box[2] < env.box_half + 0.01
    return bool(on_target and resting and env.gripper_position() < 0.5)
