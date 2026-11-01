"""学習のロス(openpi の学習ログ)と、検証データのロス(openpi_val.py の val.json)を同じ図に描く。

学習ロスは openpi が log_interval(50 ステップ)ごとに、その間の平均を "Step N: grad_norm=..., loss=..." の形で書く。
検証データ・学習データの指標(openpi_val.py の metrics_<split>.json)は、同じ損失を雑音と時刻を固定して、画像の拡張なしで計算したもの。
学習データ側も同じ方法で測るので、検証データとの差が過学習の目安になる(学習ログのロスは拡張と乱数が入るので参考)。
act_mse は行動を出させたときの誤差で、別の量なので別のパネルにする。
使い方: python scripts/plot_loss.py [--log outputs/train/run1.log --val outputs/train/metrics_val.json --train-metrics outputs/train/metrics_train.json]
"""
import argparse
import json
import os
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
    ap.add_argument("--val", default="outputs/train/metrics_val.json")
    ap.add_argument("--train-metrics", default="outputs/train/metrics_train.json", help="無ければ描かない")
    ap.add_argument("--out", default="outputs/train/loss_curve.png")
    ap.add_argument("--smooth", type=int, default=10, help="学習ロスの移動平均の点数(1 点 = 50 ステップ)")
    args = ap.parse_args()

    s, loss, gnorm = read_train(args.log)
    k = max(1, args.smooth)
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    s_smooth = s[k - 1:]

    def load(path):
        """{step: {loss, act_mse}} → ステップ(学習前 = 0)と 2 つの指標の配列。"""
        m = json.load(open(path))
        rows = sorted(((0 if key == "base" else int(key)), v) for key, v in m.items())
        return (np.array([r[0] for r in rows]), np.array([r[1]["loss"] for r in rows]),
                np.array([r[1]["act_mse"] for r in rows]))

    series = [("validation", VAL_C, "o", load(args.val))]
    if args.train_metrics and os.path.exists(args.train_metrics):
        series.insert(0, ("train data (same fixed eval)", TRAIN_C, "s", load(args.train_metrics)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        style(ax)
    a1.plot(s_smooth, smooth, color=TRAIN_C, lw=1.5, alpha=0.35, label=f"training log (mean of {k * 50} steps)")
    for name, c, mk, (st, lo, ms) in series:
        kw = dict(color=c, lw=2, marker=mk, ms=8, mec="#fcfcfb", mew=2, label=name)
        a1.plot(st, lo, **kw)
        a2.plot(st, ms, **kw)
        if name == "validation":
            i = int(np.argmin(np.where(st > 0, lo, np.inf)))
            a1.annotate(f"min {lo[i]:.4f} @ {st[i]}", (st[i], lo[i]), xytext=(0, 14), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
            print(f"val: loss min {lo[i]:.4f} @ {st[i]}, last {lo[-1]:.4f}; act_mse min {ms[st > 0].min():.4f} "
                  f"@ {st[st > 0][np.argmin(ms[st > 0])]}, last {ms[-1]:.4f}")
        else:
            print(f"train subset: loss last {lo[-1]:.4f}; act_mse last {ms[-1]:.4f}")
    for ax, yl, t in ((a1, "flow-matching loss (log)", "Loss"), (a2, "action MSE (normalized, first 8 steps, log)", "Action error")):
        ax.set_yscale("log")
        ax.set_xlabel("training step (0 = before fine-tuning)", color=MUTED)
        ax.set_ylabel(yl, color=MUTED)
        ax.set_title(t, loc="left", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print("saved", args.out)
    print(f"train loss: first {loss[0]:.4f} (step {s[0]}), last-500-step mean {loss[s > s[-1] - 500].mean():.4f}")
    print(f"grad_norm: median {np.nanmedian(gnorm):.3f}, max {np.nanmax(gnorm):.3f}")


if __name__ == "__main__":
    main()
