"""1 つの配置(条件・seed)について、学習前 | 学習後 | 教師 の 3 列 × 外部カメラ | 手首カメラ の 2 行を並べた動画を作る(提出動画の部品)。

元の動画は eval_vla.py / teacher_eval.py の出力(左半分が外部カメラ、右半分が手首カメラ、各 320x180)。
各マスを 2 倍に拡大する。上に字幕(英語)、各列の上に結果を書く。短い試行は最後の画面で止める。
使い方: python scripts/make_compare_clip.py --condition base --seed 500001 --caption "Same scene, before / after fine-tuning"
"""
import argparse
import contextlib
import json
import os

import cv2
import mediapy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SCALE, CW, CH = 2, 320, 180
GAP, CAP_H, COL_H, ROW_W = 8, 56, 44, 0
TAIL = 15     # 手を上げた後に見せるフレーム数(1 秒)
BG, INK, MUTED, OK, NG = (252, 252, 251), (11, 11, 11), (82, 81, 78), (12, 120, 12), (176, 40, 40)
COLUMNS = (("zero-shot (pi05_droid)", "base"), ("fine-tuned (LoRA, 240 demos)", "run1_s14999_strict"),
           ("teacher (scripted, privileged state)", "teacher_strict"))
# 結果は厳しい定義(置いて、離して、5 cm 以上上げる)で書く。判定し直した結果(rejudge_strict.py)があればそれを使う
OUTCOME = {"placed_not_released": "placed, not released", "no_grasp": "no grasp", "misplaced": "placed off target",
           "wrong_marker": "wrong marker", "holding": "still holding", "dropped": "dropped"}


def result(root, tag, cond, seed):
    for name in ("results_rejudged.jsonl", "results.jsonl"):
        path = os.path.join(root, tag, name)
        if not os.path.exists(path):
            continue
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if r["condition"] == cond and r["seed"] == seed:
                    return {**r, **{k: r["strict"][k] for k in ("success", "outcome", "dist_to_target")}} if "strict" in r else r
        return None
    return None


def make_clip(condition, seed, caption="", root="outputs/eval", out=None, follow=(0, 1, 2)):
    """follow: 動画の長さを決める列(既定は全部)。(1, 2) なら学習後と教師が終わった時点で切る(学習前は途中で止まる)。"""
    out = out or f"outputs/video/clips/{condition}_s{seed}.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    cw, ch = CW * SCALE, CH * SCALE
    W = 3 * cw + 2 * GAP
    H = CAP_H + COL_H + 2 * ch + GAP
    H += H % 2
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    prompt = None
    cols, limits = [], []     # 成功した列は、手を上げた時点から TAIL フレームで止める(その後また手を下ろすことがある)
    for ci, (title, tag) in enumerate(COLUMNS):
        r = result(root, tag, condition, seed)
        x = ci * (cw + GAP)
        d.text((x + 10, CAP_H + 2), title, font=ImageFont.truetype(FONT_B, 18), fill=MUTED)
        if r is None:
            d.text((x + 10, CAP_H + 22), "(not run)", font=ImageFont.truetype(FONT, 15), fill=MUTED)
        else:
            prompt = prompt or r["task"]["prompt"]
            ok = r["success"]
            txt = f"{'SUCCESS' if ok else OUTCOME.get(r['outcome'], r['outcome'].replace('_', ' '))}  ({r['dist_to_target'] * 100:.1f} cm from goal)"
            d.text((x + 10, CAP_H + 22), txt, font=ImageFont.truetype(FONT_B, 15), fill=OK if ok else NG)
        path = os.path.join(root, tag, f"{condition}_s{seed}.mp4")
        cols.append(path if os.path.exists(path) else None)
        t_up = r.get("strict", {}).get("t") if r and r["success"] else None
        limits.append(t_up + TAIL if t_up else None)
    d.text((10, 6), caption, font=ImageFont.truetype(FONT_B, 22), fill=INK)
    d.text((10, 32), f'instruction: "{prompt}"', font=ImageFont.truetype(FONT, 16), fill=MUTED)
    canvas0 = np.asarray(base).copy()

    with contextlib.ExitStack() as stack:
        readers, T = [], 1
        for p in cols:
            if p is None:
                readers.append(None)
                continue
            rd = stack.enter_context(mediapy.VideoReader(p))
            if len(readers) in follow:
                T = max(T, min(rd.num_images, limits[len(readers)] or rd.num_images))
            readers.append(iter(rd))
        last = [None] * 3
        with mediapy.VideoWriter(out, shape=(H, W), fps=15) as w:
            for fi in range(T):
                f = canvas0.copy()
                for ci, it in enumerate(readers):
                    if it is None:
                        continue
                    if limits[ci] is None or fi < limits[ci]:
                        last[ci] = next(it, last[ci])
                    x = ci * (cw + GAP)
                    for ri in range(2):      # 0: 外部カメラ(左半分)、1: 手首カメラ(右半分)
                        cell = last[ci][:, ri * CW:(ri + 1) * CW]
                        y = CAP_H + COL_H + ri * (ch + GAP)
                        f[y:y + ch, x:x + cw] = cv2.resize(cell, (cw, ch), interpolation=cv2.INTER_LINEAR)
                w.add_image(f)
    print(f"saved {out}: {T} frames ({T / 15:.0f} s), {W}x{H}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--caption", default="")
    ap.add_argument("--root", default="outputs/eval")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    make_clip(args.condition, args.seed, args.caption, args.root, args.out)


if __name__ == "__main__":
    main()
