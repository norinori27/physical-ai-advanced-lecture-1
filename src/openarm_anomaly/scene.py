"""運搬タスク用のシーンを OpenArm v1 単腕モデルから組み立てる。

MJX 向けに、衝突用の高頂点メッシュ(指は 3 万頂点超)を削除し、
ハンドと指先パッドを箱プリミティブで置き換える。腕リンクは衝突を持たない。
"""
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
OPENARM_XML = ROOT / "openarm_mujoco" / "v1" / "openarm.xml"

# 腕の土台を載せる台座の高さ。手首の可動域(joint6 ±45°)が狭く、
# 机と同じ高さに置くと上からの把持で届く範囲がほぼ無くなるため。
PEDESTAL_H = 0.15
# ステーション(机上面での物体中心の xy)。上からの把持で到達できる範囲から選んだ。
STATION_A = (0.30, -0.20)
STATION_B = (0.30, 0.20)

# 指パッド: 指ボディ座標系(z が指の伸びる向き、y が閉じる向き)。
# メッシュの内側面は y=+0.0058(右指)、先端は z=0.0805。
PAD_HALF = (0.012, 0.004, 0.015)
PAD_POS_Z = 0.065
PAD_INNER_Y = 0.0058


def build_spec(box_half=0.03, box_mass=0.5, friction=1.0, timestep=0.002):
    spec = mujoco.MjSpec.from_file(str(OPENARM_XML))
    spec.option.timestep = timestep
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

    for g in list(spec.geoms):
        if g.name.endswith("_collision"):
            spec.delete(g)
    for mesh in list(spec.meshes):
        if mesh.name.endswith("_collision"):
            spec.delete(mesh)

    # 実機のドライバと同様に重力補償する。補償トルクはアクチュエータを通るので
    # actuator_force に含まれ、トルク上限の対象にもなる。
    base = spec.body("openarm_link0")
    base.pos = [0, 0, PEDESTAL_H]
    for b in spec.bodies:
        if b.name.startswith("openarm_"):
            b.gravcomp = 1.0
    for j in spec.joints:
        if j.name.startswith("openarm_"):
            j.actgravcomp = True

    hand = spec.body("openarm_hand")
    hand.add_geom(name="hand_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.0285, 0.06, 0.013], pos=[0, 0, 0.0074],
                  rgba=[1, 0.3, 0.1, 0.3], group=3)
    hand.add_site(name="tcp", pos=[0, 0, 0.015 + PAD_POS_Z], size=[0.008] * 3,
                  rgba=[0, 1, 0, 1], group=4)
    for body, sign in (("openarm_right_finger", 1), ("openarm_left_finger", -1)):
        spec.body(body).add_geom(
            name=f"{body}_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PAD_HALF),
            pos=[0, sign * (PAD_INNER_Y - PAD_HALF[1]), PAD_POS_Z],
            friction=[friction, 0.01, 0.001], condim=4, priority=1,
            solref=[0.01, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
            rgba=[0.1, 0.8, 0.1, 1], group=3)

    world = spec.worldbody
    world.add_light(pos=[0, 0, 2], dir=[0, 0, -1], diffuse=[0.8] * 3)
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01],
                   rgba=[0.75, 0.7, 0.62, 1])
    world.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.06, PEDESTAL_H / 2, 0], pos=[0, 0, PEDESTAL_H / 2],
                   contype=0, conaffinity=0, rgba=[0.3, 0.3, 0.32, 1])
    world.add_camera(name="overview", pos=[1.1, 0, 0.75], xyaxes=[0, 1, 0, -0.5, 0, 0.87])
    for name, (x, y) in (("station_a", STATION_A), ("station_b", STATION_B)):
        world.add_site(name=name, pos=[x, y, 0.0005], size=[0.05, 0.05, 0.0005],
                       type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[0.2, 0.4, 0.9, 0.4])

    box = world.add_body(name="box", pos=[*STATION_A, box_half])
    box.add_freejoint(name="box_free")
    box.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[box_half] * 3,
                 mass=box_mass, friction=[friction, 0.01, 0.001], condim=4,
                 rgba=[0.85, 0.55, 0.2, 1])

    # home キーフレームに箱の qpos を足す
    key = spec.keys[0]
    key.qpos = list(key.qpos) + [*STATION_A, box_half, 1, 0, 0, 0]
    return spec


def build_model(**kw):
    return build_spec(**kw).compile()
