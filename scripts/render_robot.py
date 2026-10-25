"""Day 1: ロボットの見た目、元の衝突メッシュ、置き換えた衝突箱を画像にする。"""
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, "src")
from openarm_anomaly.scene import OPENARM_XML, build_model
from openarm_anomaly.scripted import reset_to_ready

OUT = "outputs/day1"


def render(m, d, groups, lookat, dist, az, el, size=(480, 640)):
    r = mujoco.Renderer(m, *size)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in groups:
        opt.geomgroup[g] = 1
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
    r.update_scene(d, camera=cam, scene_option=opt)
    img = r.render()
    r.close()
    return img


def label(img, text):
    import PIL.ImageDraw
    im = Image.fromarray(img)
    PIL.ImageDraw.Draw(im).text((10, 10), text, fill=(0, 0, 0))
    return np.asarray(im)


# 1) 作業シーン全体(準備姿勢)
m = build_model(box_half=0.03)
d = mujoco.MjData(m)
reset_to_ready(m, d, 0.03)
scene = render(m, d, [0, 2], [0.2, 0, 0.2], 1.4, 150, -25)
hand = d.body("openarm_hand").xpos.copy()

# 2) グリッパ拡大: 見た目 / 置き換えた衝突箱(見た目を半透明に重ねる)
vis = render(m, d, [2], hand + [0, 0, -0.05], 0.35, 150, -15)
m.geom_rgba[[g for g in range(m.ngeom) if m.geom_group[g] == 2], 3] = 0.25
box = render(m, d, [2, 3], hand + [0, 0, -0.05], 0.35, 150, -15)

# 3) 元モデルの衝突メッシュ(home 姿勢のグリッパ拡大)
m0 = mujoco.MjModel.from_xml_path(str(OPENARM_XML))
d0 = mujoco.MjData(m0)
mujoco.mj_resetDataKeyframe(m0, d0, 0)
mujoco.mj_forward(m0, d0)
h0 = d0.body("openarm_hand").xpos.copy()
orig_vis = render(m0, d0, [2], h0 + [0.04, 0, 0], 0.35, 135, -20)
orig_col = render(m0, d0, [3], h0 + [0.04, 0, 0], 0.35, 135, -20)

Image.fromarray(scene).save(f"{OUT}/scene.png")
Image.fromarray(np.hstack([label(orig_vis, "original: visual mesh"),
                           label(orig_col, "original: collision mesh (finger 32k verts)")])).save(f"{OUT}/gripper_original.png")
Image.fromarray(np.hstack([label(vis, "ours: visual mesh"),
                           label(box, "ours: collision boxes (green pads, orange hand)")])).save(f"{OUT}/gripper_ours.png")
print("saved")
