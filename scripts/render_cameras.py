"""VLA 入力用カメラ(exterior / wrist)と全景の静止画を 1 枚にまとめる。
使い方: MUJOCO_GL=osmesa python scripts/render_cameras.py [out.png]
上段: 高画質(間引きなし・影あり)の全景と外部カメラ。下段: VLA に渡す 224x224(間引き・影なし)を 2 倍に拡大。"""
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "src")
from openarm_anomaly.layout import PICK
from openarm_anomaly.light_mesh import lighten_visual_meshes
from openarm_anomaly.scene import build_spec
from openarm_anomaly.scripted import reset_to_ready

out = sys.argv[1] if len(sys.argv) > 1 else "outputs/day3/cameras.png"
half = 0.035
others = PICK[1:]


def shots(light, cams, size):
    spec = build_spec(box_half=half, timestep=0.004, box_xy=PICK[0], obstacles=others)
    if light:
        lighten_visual_meshes(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    reset_to_ready(m, d, half, PICK[0])
    if light:
        m.light_castshadow[:] = 0
    r = mujoco.Renderer(m, *size)
    imgs = []
    for c in cams:
        r.update_scene(d, camera=c)
        imgs.append(r.render().copy())
    r.close()  # osmesa は同時に 2 つ作ると砂嵐になる
    return imgs


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(im)


hi = shots(False, ["overview", "exterior"], (448, 448))
lo = shots(True, ["exterior", "wrist"], (224, 224))
lo = [np.kron(x, np.ones((2, 2, 1), dtype=np.uint8)) for x in lo]
top = np.concatenate([label(hi[0], "overview (high quality)"), label(hi[1], "exterior (high quality)")], axis=1)
bot = np.concatenate([label(lo[0], "exterior 224 (VLA input)"), label(lo[1], "wrist 224 (VLA input)")], axis=1)
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
Image.fromarray(np.concatenate([top, bot], axis=0)).save(out)
print("saved", out)
