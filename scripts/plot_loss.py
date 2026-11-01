"""学習のロス(openpi の学習ログ)と、検証データのロス(openpi_val.py の val.json)を同じ図に描く。

学習ロスは openpi が log_interval(50 ステップ)ごとに、その間の平均を "Step N: grad_norm=..., loss=..." の形で書く。
検証ロスは同じ損失(流れのマッチングの誤差)を、雑音と時刻を固定して計算したもの。act_mse は行動を出させたときの誤差で、別の量なので別のパネルにする。
使い方: python scripts/plot_loss.py --log outputs/train/run1.log --val outputs/train/val.json --out outputs/train/loss_curve.png
(ログと val.json は、ホストで modal volume get openarm-ckpt logs/run1.log outputs/train/ などで取得する)
"""
import argparse
import json
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAIN_C, VAL_C = "#2a78d6", "#eb6834"     # 色の並び 1・2(青・オレンジ)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_train(path):
    steps, loss, gnorm = [], [], []
    for line in open(path, errors="ignore"):
        m = re.search(r"Step (\d+): (.*)", line)
        if not m:
            continue
        kv = dict(re.findall(r"(\w+)=([-\d.eE+]+)", m.group(2)))
        if "loss" in kv:
            steps.append(int(m.group(1)))
            loss.append(float(kv["loss"]))
            gnorm.append(float(kv.get("grad_norm", "nan")))
    # 再開などで同じステップが 2 回書かれていたら、後の方を使う
    d = {s: (l, g) for s, l, g in zip(steps, loss, gnorm)}
    s = np.array(sorted(d))
    return s, np.array([d[k][0] for k in s]), np.array([d[k][1] for k in s])


def style(ax):
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="outputs/train/run1.log")
    ap.add_argument("--val", default="outputs/train/val.json")
    ap.add_argument("--out", default="outputs/train/loss_curve.png")
    ap.add_argument("--smooth", type=int, default=10, help="学習ロスの移動平均の点数(1 点 = 50 ステップ)")
    args = ap.parse_args()

    s, loss, gnorm = read_train(args.log)
    val = json.load(open(args.val))
    base = val.pop("base", None)
    vs = np.array(sorted(int(k) for k in val))
    vloss = np.array([val[str(k)]["loss"] for k in vs])
    vmse = np.array([val[str(k)]["act_mse"] for k in vs])
    k = max(1, args.smooth)
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    s_smooth = s[k - 1:]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        style(ax)

    a1.plot(s, loss, color=TRAIN_C, lw=1, alpha=0.25)
    a1.plot(s_smooth, smooth, color=TRAIN_C, lw=2, label=f"train (mean of {k * 50} steps)")
    if base:     # 学習前(ステップ 0)の検証ロスもつなげる
        vs_all, vloss_all = np.r_[0, vs], np.r_[base["loss"], vloss]
    else:
        vs_all, vloss_all = vs, vloss
    a1.plot(vs_all, vloss_all, color=VAL_C, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2, label="validation (33 held-out demos)")
    i = int(np.argmin(vloss))
    a1.annotate(f"min {vloss[i]:.4f} @ {vs[i]}", (vs[i], vloss[i]), xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=9, color=INK)
    a1.set_yscale("log")
    a1.set_xlabel("training step", color=MUTED)
    a1.set_ylabel("flow-matching loss (log scale)", color=MUTED)
    title = "Loss" + (f"   (before fine-tuning: val {base['loss']:.3f})" if base else "")
    a1.set_title(title, loc="left", color=INK, fontsize=11)
    a1.legend(frameon=False, fontsize=9, labelcolor=INK)

    if base:
        vs2, vmse = np.r_[0, vs], np.r_[base["act_mse"], vmse]
    else:
        vs2 = vs
    a2.plot(vs2, vmse, color=VAL_C, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2)
    a2.set_xlabel("training step", color=MUTED)
    a2.set_ylabel("action MSE (normalized, first 8 steps, log)", color=MUTED)
    t2 = "Validation action error" + (f"   (before: {base['act_mse']:.3f})" if base else "")
    a2.set_title(t2, loc="left", color=INK, fontsize=11)
    a2.set_yscale("log")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print("saved", args.out)
    print(f"train loss: first {loss[0]:.4f} (step {s[0]}), last-500-step mean {loss[s > s[-1] - 500].mean():.4f}")
    print(f"val loss: min {vloss.min():.4f} @ {vs[np.argmin(vloss)]}, last {vloss[-1]:.4f}")
    print(f"grad_norm: median {np.nanmedian(gnorm):.3f}, max {np.nanmax(gnorm):.3f}")


if __name__ == "__main__":
    main()
