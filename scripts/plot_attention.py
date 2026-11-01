"""解析(analyze_policy.py の attention)の注目度の地図を、カメラの画像に重ねる。ゴールのマーカーを見ているかを数字にする。

注目度: 行動を出す部分(action expert)のトークンから、画像の 16x16 の区画への注目(層 × 雑音除去ステップの平均)。
  モデルの入力は 320x180 の画像を 224x224 に縮めて上下に黒帯を足したもの(resize_with_pad)。黒帯の分を除いて元の画像に戻す。
物の場所: 画像の色で見分ける(ゴールのマーカー、もう一方のマーカー、箱)。照明は白なので、色度(r, g, b の比)で近い色に分ける。
数字: 各物への注目の割合(画像 1 枚の注目の合計に対して)と、その物の面積の割合。割合 ÷ 面積 が 1 より大きければ、
  面積から期待されるより強く注目している(濃縮度)。
注意: 注目度は「何に頼って行動を決めたか」そのものではない(Jain & Wallace, 2019)。カメラを隠す・指示を入れ替える結果と合わせて読む。
使い方: python scripts/plot_attention.py --analysis outputs/analysis/run1_s14999_test/analysis.json
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "src")
from openarm_anomaly.task import BOX_RGB, MARKER_RGB  # noqa: E402

IMG_H, IMG_W, PAD = 180, 320, 224
OBJ_COLOR = {"goal": "#e8ff00", "other marker": "#00e5ff", "cube": "#ff00d4"}


def attention_on_image(att16):
    """16x16 の注目度(224x224 の入力に対応)を、元の 320x180 の画像の上の値に戻す(合計 1)。"""
    a = cv2.resize(np.asarray(att16, np.float32), (PAD, PAD), interpolation=cv2.INTER_NEAREST)
    h = round(PAD * IMG_H / IMG_W)
    top = (PAD - h) // 2
    a = cv2.resize(a[top:top + h], (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    return a / a.sum()


def object_masks(img, task):
    """色度で、ゴールのマーカー、もう一方のマーカー、箱の画素を見分ける。"""
    other = [c for c, _ in task["markers"] if c != task["target"]]
    refs = {"goal": MARKER_RGB[task["target"]], "cube": BOX_RGB[task["box_color"]]}
    if other:
        refs["other marker"] = MARKER_RGB[other[0]]
    x = img.astype(np.float32) + 1
    chroma = x / x.sum(-1, keepdims=True)
    sat = (x.max(-1) - x.min(-1)) / x.max(-1)
    names = list(refs)
    ref = np.array([np.asarray(refs[n]) / sum(refs[n]) for n in names])
    dist = np.linalg.norm(chroma[:, :, None, :] - ref[None, None], axis=-1)
    near = dist.argmin(-1)
    ok = (sat > 0.35) & (dist.min(-1) < 0.08)
    return {n: ok & (near == i) for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--npz", default="outputs/eval_npz/run1_s14999_strict/run1_s14999_strict")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.analysis), "attention")
    os.makedirs(out, exist_ok=True)
    res = json.load(open(args.analysis))
    rows = []
    for ep in res["episodes"]:
        z = np.load(os.path.join(args.npz, ep["file"]))
        task = json.loads(str(z["task_json"]))
        frames = [(k, fr) for k, fr in ep["frames"].items() if "attention" in fr]
        fig, axes = plt.subplots(len(frames), 2, figsize=(9, 2.7 * len(frames)), squeeze=False)
        for fi, (fname, fr) in enumerate(frames):
            for ci, (cam, key) in enumerate((("ext", "ext_16x16"), ("wrist", "wrist_16x16"))):
                img = z[cam][fr["t"]]
                att = attention_on_image(fr["attention"][key])
                masks = object_masks(img, task)
                row = {"file": ep["file"], "condition": ep["condition"], "frame": fname, "t": fr["t"], "camera": cam}
                for n, mk in masks.items():
                    area = float(mk.mean())
                    share = float(att[mk].sum())
                    row[n] = {"share": share, "area": area, "enrichment": share / area if area > 0 else None}
                rows.append(row)
                ax = axes[fi, ci]
                ax.imshow(img)
                ax.imshow(att, cmap="magma", alpha=0.55, interpolation="nearest")
                label = []
                for n, mk in masks.items():
                    if mk.any():
                        ax.contour(mk.astype(float), levels=[0.5], colors=OBJ_COLOR[n], linewidths=1.2)
                        e = row[n]["enrichment"]
                        label.append(f"{n} x{e:.1f}" if e is not None else f"{n} -")
                    else:
                        label.append(f"{n}: not visible")
                ax.set_title(f"{fname} (t={fr['t']}) {cam}\n" + ", ".join(label), fontsize=8)
                ax.axis("off")
        fig.suptitle(f"{ep['file']}: \"{ep['prompt']}\"  (outline: goal yellow, other marker cyan, cube magenta;"
                     f" x = attention share / area)", fontsize=8)
        fig.tight_layout()
        path = os.path.join(out, ep["file"].replace(".npz", ".png"))
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print("saved", path)
    with open(os.path.join(out, "object_attention.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(r["file"], r["frame"], r["camera"],
              {n: (f"share {r[n]['share']:.3f} area {r[n]['area']:.3f} x{r[n]['enrichment']:.1f}"
                   if r.get(n) and r[n]["enrichment"] is not None else "not visible")
               for n in OBJ_COLOR if n in r})


if __name__ == "__main__":
    main()
