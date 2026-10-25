"""Day 1: 指パッドの位置を真横・正面から確認する。"""
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "src")
from openarm_anomaly.scene import build_model
from openarm_anomaly.scripted import reset_to_ready

m = build_model(box_half=0.03)
d = mujoco.MjData(m)
reset_to_ready(m, d, 0.03)
m.light_diffuse[:] = 1.0
vis = [g for g in range(m.ngeom) if m.geom_group[g] == 2]
m.geom_rgba[vis] = [0.6, 0.6, 0.65, 0.35]
m.geom_rgba[[m.geom(n).id for n in ("openarm_right_finger_pad", "openarm_left_finger_pad")]] = [0, 1, 0, 1]
center = (d.body("openarm_right_finger").xpos + d.body("openarm_left_finger").xpos) / 2 + [0, 0, -0.04]
R = d.body("openarm_hand").xmat.reshape(3, 3)
close_axis = R[:, 1]  # 指が閉じる向き(世界座標)
az_close = np.degrees(np.arctan2(close_axis[1], close_axis[0]))

r = mujoco.Renderer(m, 480, 480)
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[0] = opt.geomgroup[2] = opt.geomgroup[3] = 1
imgs = []
for az, title in ((az_close + 90, "side: closing axis left-right"), (az_close, "front: along closing axis")):
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = center, 0.28, az, 0
    r.update_scene(d, camera=cam, scene_option=opt)
    im = Image.fromarray(r.render())
    ImageDraw.Draw(im).text((10, 10), title, fill=(255, 255, 255))
    imgs.append(np.asarray(im))
Image.fromarray(np.hstack(imgs)).save("outputs/day1/gripper_pads_check.png")
