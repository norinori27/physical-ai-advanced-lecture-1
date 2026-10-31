"""デモデータの変数の分布と、教師の失敗の内訳を集計する。

使い方: python scripts/analyze_demos.py --src data/demos --out outputs/demo_check/demo_distribution.png
"""
import argparse
import glob
import itertools
import json
import os
from collections import Counter

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # 箱 / ゴールのマーカー / もう 1 個のマーカー


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/demos")
    ap.add_argument("--out", default="outputs/demo_check/demo_distribution.png")
    args = ap.parse_args()

    tasks, steps = [], []
    for f in sorted(glob.glob(os.path.join(args.src, "ep_*.npz"))):
        with np.load(f) as d:
            tasks.append(json.loads(str(d["task_json"])))
            steps.append(len(d["actions"]))
    n = len(tasks)
    print(f"エピソード {n} 本、ステップ数 平均 {np.mean(steps):.0f}(最小 {min(steps)}、最大 {max(steps)})、合計 {sum(steps)} フレーム")

    # 水準の数え上げ
    tmpl = lambda t: t["template"].split(" the ")[0]  # noqa: E731  put / place / move
    for name, key in (("箱の色", lambda t: t["box_color"]), ("ゴールの色", lambda t: t["target"]), ("言い回し", tmpl)):
        print(name, dict(Counter(key(t) for t in tasks)))
    combo = Counter((t["box_color"], t["target"], tmpl(t)) for t in tasks)
    levels = list(itertools.product(("orange", "red"), ("blue", "green"), ("put", "place", "move")))
    counts = [combo.get(c, 0) for c in levels]
    print(f"12 通りの組み合わせの本数: 最小 {min(counts)}、最大 {max(counts)}、平均 {np.mean(counts):.1f}")

    # 教師の失敗の内訳(summary.json がある場合)
    sp = os.path.join(args.src, "summary.json")
    if os.path.exists(sp):
        eps = json.load(open(sp))["episodes"]
        cat = Counter()
        for e in eps:
            pb = e.get("phase_box", {})
            if e["success"]:
                cat["成功"] += 1
            elif pb.get("lift", [0, 0, 0])[2] < 0.08:
                cat["つかみ損ね"] += 1
            elif pb.get("carry", [0, 0, 0])[2] < 0.08:
                cat["運搬中に落下"] += 1
            else:
                cat["置く位置のずれ"] += 1
        print("教師の試行", len(eps), "本の内訳:", dict(cat))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))

    ax = axs[0, 0]   # 組み合わせごとの本数
    ax.bar(range(12), counts, color=SERIES[0], width=0.7)
    ax.set_xticks(range(12), [f"{b[0]}{'B' if g == 'blue' else 'G'}-{t}" for b, g, t in levels], rotation=60, fontsize=7)
    ax.axhline(n / 12, color=INK2, lw=1, ls="--")
    ax.set_title("Episodes per combination (box color x goal x phrasing)", color=INK, fontsize=9, loc="left")
    ax.set_ylabel("episodes")

    ax = axs[0, 1]   # 机上の配置
    pts = {"box": [t["box_xy"] for t in tasks], "goal marker": [t["target_xy"] for t in tasks],
           "other marker": [p for t in tasks for c, p in t["markers"] if c != t["target"]]}
    for (lab, p), col in zip(pts.items(), SERIES):
        p = np.array(p)
        ax.scatter(p[:, 1], p[:, 0], s=10, color=col, label=lab, alpha=0.7, edgecolors="none")
    ax.add_patch(plt.Rectangle((-0.26, 0.18), 0.52, 0.14, fill=False, ec=INK2, ls="--", lw=1))
    ax.set_xlabel("y [m]"); ax.set_ylabel("x [m] (forward)")
    ax.set_xlim(0.38, -0.38); ax.set_ylim(0.1, 0.4)
    ax.set_title("Layout on the table (dashed = training region)", color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axs[0, 2]   # 開始時の手先
    r = np.array([t["ready_tcp"] for t in tasks])
    ax.scatter(r[:, 1], r[:, 0], s=10, c=SERIES[0], alpha=0.7, edgecolors="none")
    ax.set_xlabel("y [m]"); ax.set_ylabel("x [m]")
    ax.set_title(f"Start TCP (z {r[:, 2].min():.2f}-{r[:, 2].max():.2f} m)", color=INK, fontsize=9, loc="left")

    ax = axs[1, 0]   # 照明の明るさ
    ax.hist([t["light_diffuse"] for t in tasks], bins=16, color=SERIES[0], rwidth=0.85)
    ax.set_xlabel("light diffuse"); ax.set_ylabel("episodes")
    ax.set_title("Light intensity", color=INK, fontsize=9, loc="left")

    ax = axs[1, 1]   # 照明の位置
    lp = np.array([t["light_pos"] for t in tasks])
    ax.scatter(lp[:, 1], lp[:, 0], s=10, c=SERIES[0], alpha=0.7, edgecolors="none")
    ax.set_xlabel("y [m]"); ax.set_ylabel("x [m]")
    ax.set_title("Light position (height 2 m)", color=INK, fontsize=9, loc="left")

    ax = axs[1, 2]   # エピソードの長さ
    ax.hist(steps, bins=20, color=SERIES[0], rwidth=0.85)
    ax.set_xlabel("steps (15 Hz)"); ax.set_ylabel("episodes")
    ax.set_title("Episode length", color=INK, fontsize=9, loc="left")

    for a in axs.flat:
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
