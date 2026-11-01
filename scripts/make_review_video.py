"""評価の動画を条件ごとに並べて、見比べ用の動画を作る。

1 本の動画 = 1 つの条件。列: 学習前(zero-shot、seed 500000〜04)| 学習後(同じ seed)| 学習後(seed 500005〜09)。
各マスは評価の動画(外部カメラ | 手首カメラ)で、上に seed・結果・ゴールまでの距離・指示文を書く。短い試行は最後の画面で止める。
使い方: python scripts/make_review_video.py [--zs base --ft run1_s14999] [--out outputs/video/review] [--speed 1]
"""
import argparse
import contextlib
import json
import os
from multiprocessing import Pool

import mediapy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CELL_H, CELL_W, LABEL_H, HEAD_H = 180, 640, 34, 64
BG, INK, MUTED = (252, 252, 251), (11, 11, 11), (82, 81, 78)
OK, NG = (12, 120, 12), (176, 40, 40)
COND_TEXT = {"base": "base: same ranges as training (unseen combinations)",
             "pos_outer": "pos_outer: box / goal outside the training area",
             "goal_color": "goal_color: unseen goal color (yellow)",
             "light": "light: unseen lighting (dim / oblique)",
             "box_color": "box_color: unseen box color (purple)",
             "language": "language: unseen instruction wording"}


def load_results(root, tag):
    out = {}
    with open(os.path.join(root, tag, "results.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["condition"], r["seed"])] = r
    return out


def label(r, seed, w):
    img = Image.new("RGB", (w, LABEL_H), BG)
    d = ImageDraw.Draw(img)
    if r is None:
        d.text((8, 9), f"seed {seed}: (not evaluated)", font=ImageFont.truetype(FONT, 13), fill=MUTED)
        return np.asarray(img)
    ok = r["success"]
    head = f"seed {seed}  {'SUCCESS' if ok else r['outcome']}  ({r['dist_to_target'] * 100:.1f} cm)"
    d.text((8, 2), head, font=ImageFont.truetype(FONT_B, 13), fill=OK if ok else NG)
    d.text((8, 18), r["task"]["prompt"], font=ImageFont.truetype(FONT, 12), fill=MUTED)
    return np.asarray(img)


def make(job):
    with contextlib.ExitStack() as stack:     # 開いた VideoReader を最後にまとめて閉じる
        return _make(job, stack)


def _make(job, stack):
    cond, root, zs, ft, out, speed = job
    res = {"zs": load_results(root, zs), "ft": load_results(root, ft)}
    cols = [("zero-shot (before fine-tuning)", "zs", range(500000, 500005)),
            ("fine-tuned (same seeds)", "ft", range(500000, 500005)),
            ("fine-tuned (more seeds)", "ft", range(500005, 500010))]
    # 元の動画は丸ごと読まず、1 フレームずつ読む(15 本を丸ごと読むと 1 プロセスで約 2.3 GB。
    # 11-01 に 5 並列で動かして WSL ごと落ちた)
    cells = []      # (列, 行, フレームのイテレータ or None, ラベル)
    T = 1
    for ci, (_, key, seeds) in enumerate(cols):
        for ri, seed in enumerate(seeds):
            path = os.path.join(root, zs if key == "zs" else ft, f"{cond}_s{seed}.mp4")
            r = res[key].get((cond, seed))
            v = None
            if os.path.exists(path):
                reader = stack.enter_context(mediapy.VideoReader(path))
                n = reader.num_images
                T = max(T, (n + speed - 1) // speed)
                v = (img for i, img in enumerate(reader) if i % speed == 0)
            cells.append((ci, ri, v, label(r, seed, CELL_W)))
    W, H = CELL_W * 3 + 2 * 8, HEAD_H + 5 * (CELL_H + LABEL_H)
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    d.text((10, 6), COND_TEXT.get(cond, cond), font=ImageFont.truetype(FONT_B, 20), fill=INK)
    n_zs = sum(r["success"] for (c, _), r in res["zs"].items() if c == cond)
    n_ft = sum(r["success"] for (c, _), r in res["ft"].items() if c == cond)
    m_zs = sum(c == cond for c, _ in res["zs"])
    m_ft = sum(c == cond for c, _ in res["ft"])
    for ci, (title, key, _) in enumerate(cols):
        x = ci * (CELL_W + 8)
        extra = f"   [{n_zs}/{m_zs}]" if key == "zs" else (f"   [{n_ft}/{m_ft} total]" if ci == 1 else "")
        d.text((x + 8, 38), title + extra, font=ImageFont.truetype(FONT_B, 15), fill=MUTED)
    canvas0 = np.asarray(base).copy()
    for ci, ri, _, lab in cells:
        x, y = ci * (CELL_W + 8), HEAD_H + ri * (CELL_H + LABEL_H)
        canvas0[y:y + LABEL_H, x:x + CELL_W] = lab
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"review_{cond}.mp4")
    last = {}       # 終わった試行は最後の画面で止める
    with mediapy.VideoWriter(path, shape=(H, W), fps=15) as w:     # 1 フレームずつ書く(全部溜めると数 GB になる)
        for t in range(T):
            f = canvas0.copy()
            for ci, ri, v, _ in cells:
                if v is None:
                    continue
                last[(ci, ri)] = next(v, last.get((ci, ri)))
                x, y = ci * (CELL_W + 8), HEAD_H + ri * (CELL_H + LABEL_H) + LABEL_H
                f[y:y + CELL_H, x:x + CELL_W] = last[(ci, ri)]
            w.add_image(f)
    return f"{path}: {T} frames ({T / 15:.0f} s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/eval")
    ap.add_argument("--zs", default="base")
    ap.add_argument("--ft", default="run1_s14999")
    ap.add_argument("--out", default="outputs/video/review")
    ap.add_argument("--speed", type=int, default=1, help="n 倍速(n フレームおきに使う)")
    ap.add_argument("--conditions", nargs="+", default=list(COND_TEXT))
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    jobs = [(c, args.root, args.zs, args.ft, args.out, args.speed) for c in args.conditions]
    # 1 本で ffmpeg を 16 個(読み 15 + 書き 1)動かすので、並列は少なめにする
    with Pool(min(args.jobs, len(jobs))) as p:
        for line in p.imap_unordered(make, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
