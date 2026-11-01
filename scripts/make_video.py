"""提出動画を組み立てる(場面の構成は report/video_plan.md)。

各場面を 1936x828 の画面にそろえ、つなげて 1 本にする。説明は映像の上の英語の字幕だけ(スライドは入れない)。
  clip  : make_compare_clip.py の 2x3(学習前 | 学習後 | 教師 × 外部 | 手首)。学習後と教師が終わった時点で切る
  grid  : 6 条件の学習後の外部カメラを 3x2 に並べ、各マスに成功数
--speed で再生速度を変える(フレームを間引く)。字幕に速度を書く。
使い方: python scripts/make_video.py [--speed 1.5] [--out outputs/video/openarm_vla.mp4]
"""
import argparse
import contextlib
import os
import sys

import cv2
import mediapy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "scripts")
from make_compare_clip import BG, FONT, FONT_B, INK, MUTED, OK, NG, TAIL, make_clip, result  # noqa: E402

W, H, FPS = 1936, 828, 15
FT = "run1_s14999_strict"
# (種類, 引数, 字幕)。成功数は 2 回目の評価(厳しい定義: 置いて、離して、5 cm 以上上げる)
SCENES = [
    ("clip", ("base", 500002), "Same scene, trained-range layout: fine-tuned 4/10, teacher 7/10 (zero-shot 0/5)"),
    ("clip", ("base", 500003), "Typical failure: the cube is carried to the goal but placed off the marker"),
    ("clip", ("goal_color", 500002), "Unseen goal color (yellow): fine-tuned 2/10, teacher 8/10"),
    ("clip", ("box_color", 500001), "Unseen cube color (purple): fine-tuned 5/10, teacher 7/10"),
    ("clip", ("light", 500004), "Unseen lighting (dark / oblique): fine-tuned 2/10, teacher 7/10"),
    ("clip", ("language", 500001), "Unseen wording: fine-tuned 3/10, teacher 7/10"),
    ("clip", ("pos_outer", 500001), "Outside the trained area: fine-tuned 1/10, teacher also 2/10 (edge of reach)"),
    ("clip", ("pos_outer", 500006), "Outside the trained area: a fine-tuned success where the teacher fails"),
    ("clip", ("base", 500001), "Failure: the cube slips out of the gripper while carrying"),
    ("grid", [("base", 500002, "4/10"), ("pos_outer", 500006, "1/10"), ("goal_color", 500002, "2/10"),
              ("light", 500004, "2/10"), ("box_color", 500001, "5/10"), ("language", 500001, "3/10")],
     "Fine-tuned with 240 demos (LoRA): 17/60 overall vs zero-shot 0/30 (place, release and lift)"),
]


def text_canvas(caption, sub="", speed=1.0):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((10, 6), caption, font=ImageFont.truetype(FONT_B, 22), fill=INK)
    if speed != 1.0:
        d.text((W - 90, 8), f"x{speed:g}", font=ImageFont.truetype(FONT_B, 20), fill=MUTED)
    if sub:
        d.text((10, 32), sub, font=ImageFont.truetype(FONT, 16), fill=MUTED)
    return img, d


def clip_frames(cond, seed, caption, speed):
    path = make_clip(cond, seed, caption, out=f"outputs/video/clips/{cond}_s{seed}.mp4", follow=(1, 2))
    with mediapy.VideoReader(path) as rd:
        for fr in rd:
            if speed != 1.0:
                fr = fr.copy()
                cv2.putText(fr, f"x{speed:g}", (W - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, MUTED, 2, cv2.LINE_AA)
            yield fr


def grid_frames(items, caption, speed):
    img, d = text_canvas(caption, "", speed)
    cw, ch, gap, lab = 640, 360, 8, 26
    for k, (cond, seed, n) in enumerate(items):
        x, y = (k % 3) * (cw + gap), 48 + (k // 3) * (ch + lab + gap)
        d.text((x + 6, y + 2), f"{cond}: {n}", font=ImageFont.truetype(FONT_B, 18), fill=INK)
    canvas0 = np.asarray(img).copy()
    with contextlib.ExitStack() as stack:
        readers, limits, T = [], [], 1
        for cond, seed, _ in items:
            rd = stack.enter_context(mediapy.VideoReader(os.path.join("outputs/eval", FT, f"{cond}_s{seed}.mp4")))
            r = result("outputs/eval", FT, cond, seed)
            t_up = r.get("strict", {}).get("t") if r and r["success"] else None
            limits.append(min(rd.num_images, t_up + TAIL) if t_up else rd.num_images)
            readers.append(iter(rd))
        T = max(limits)
        last = [None] * len(items)
        for fi in range(T):
            f = canvas0.copy()
            for k, it in enumerate(readers):
                if fi < limits[k]:
                    last[k] = next(it, last[k])
                x, y = (k % 3) * (cw + gap), 48 + (k // 3) * (ch + lab + gap) + lab
                f[y:y + ch, x:x + cw] = cv2.resize(last[k][:, :320], (cw, ch), interpolation=cv2.INTER_LINEAR)
            yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--hold", type=float, default=1.0, help="各場面の最後の画面で止める秒数")
    ap.add_argument("--out", default="outputs/video/openarm_vla.mp4")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total = 0
    with mediapy.VideoWriter(args.out, shape=(H, W), fps=FPS) as w:
        for kind, arg, caption in SCENES:
            gen = {"clip": lambda: clip_frames(*arg, caption, args.speed),
                   "grid": lambda: grid_frames(arg, caption, args.speed)}[kind]()
            n, acc, last = 0, 0.0, None
            for fr in gen:          # 速度 s なら、s フレームごとに 1 枚を書く
                last = fr
                acc += 1.0
                if acc >= args.speed:
                    acc -= args.speed
                    w.add_image(fr)
                    n += 1
            for _ in range(int(args.hold * FPS)):
                w.add_image(last)
            n += int(args.hold * FPS)
            total += n
            print(f"{kind} {arg if kind != 'grid' else ''}: {n} frames ({n / FPS:.1f} s)", flush=True)
    print(f"saved {args.out}: {total / FPS:.0f} s at x{args.speed:g}")


if __name__ == "__main__":
    main()
