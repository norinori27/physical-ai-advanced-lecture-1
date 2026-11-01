"""eval_vla.py の results.jsonl を条件ごとに集計し、表(Markdown)・JSON・図にする。

  成功率      : Wilson の 95% 信頼区間つき(本数が少ないので区間を必ず添える)
  到達段階    : つかめなかった / つかんだが持ち上げず / 持ち上げたが落とした・持ったまま / 置いたがずれた / 成功
  ゴールまでの距離 : 箱の最終位置とゴールの中心の距離(中央値)
  対応のある比較 : base と同じ seed の試行を 1 本ずつ比べる(pos_outer は位置そのものを変えるので除く)。
                  成功の食い違いの本数(base だけ成功 / 条件だけ成功)と、McNemar の正確検定(二項検定)の p 値。
                  注意: 同じ seed でも、条件によって乱数を引く回数が違うので、揃わない項目がある(評価の seed 500000〜09 で確認):
                    goal_color: ゴールに選ぶマーカーが違う(ゴールの位置が base と違うのは 10 本中 4 本)
                    light     : 箱のヨーのずれ(±7°)が違う
                    box_color : 指示の言い回しが違う(学習した 3 通りの中で。10 本中 9 本)
                    language  : 指示の言い回しだけが違う(= 条件そのもの)
                  ゴールの位置と開始時の手先が base と一致した本数も表示する
使い方: python scripts/summarize_eval.py --tags base run1_s14999 [--root outputs/eval] [--out outputs/eval/summary]
"""
import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONDITIONS = ("base", "pos_outer", "goal_color", "light", "box_color", "language")
# 到達段階(順序のある分類)。色は 1 色の濃淡(薄い = 手前の段階で失敗、濃い = 成功に近い)
STAGES = ("no_grasp", "grasped", "lifted", "placed_off", "placed_not_released", "success")
STAGE_LABEL = {"no_grasp": "no grasp", "grasped": "grasped, not lifted", "lifted": "lifted, dropped / still holding",
               "placed_off": "placed off target", "placed_not_released": "placed, hand not lifted away",
               "success": "success"}
STAGE_COLOR = {"no_grasp": "#cde2fb", "grasped": "#9ec5f4", "lifted": "#6da7ec", "placed_off": "#3987e5",
               "placed_not_released": "#1c5cab", "success": "#0d366b"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def tag_label(tag):
    """図のタイトル。タグ base(学習前)は条件の base と紛らわしいので言い換える。"""
    if tag == "base":
        return "zero-shot (pi05_droid)"
    if tag.startswith("run") and "_s" in tag:
        return f"fine-tuned ({tag.split('_s')[0]}, step {tag.split('_s')[1]})"
    return tag


def stage(r):
    o = r["outcome"]
    if o == "success":
        return "success"
    if o == "placed_not_released":
        return "placed_not_released"
    if o in ("misplaced", "wrong_marker") and r.get("lifted"):
        return "placed_off"
    if r.get("lifted") or o in ("dropped", "holding"):
        return "lifted"
    if o == "no_grasp":
        return "no_grasp"
    return "grasped"     # つかんだが持ち上げずに離した、など


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """食い違い b(base だけ成功)と c(条件だけ成功)の両側の正確検定。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(root, tag, strict=False, results="results.jsonl"):
    """strict: 厳しい定義(置いて、離して、上げる)の結果があれば、それで成功・結果・距離を置き換える。"""
    rows = {}
    with open(os.path.join(root, tag, results)) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if strict and "strict" in r:
                    r = {**r, "success": r["strict"]["success"], "outcome": r["strict"]["outcome"],
                         "dist_to_target": r["strict"]["dist_to_target"]}
                rows[(r["condition"], r["seed"])] = r     # 同じ試行が 2 回あれば後の方
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--root", default="outputs/eval")
    ap.add_argument("--out", default="outputs/eval/summary")
    ap.add_argument("--results", default="results.jsonl", help="読むファイル名(例: results_rejudged.jsonl)。無いタグは results.jsonl")
    ap.add_argument("--strict", action="store_true", help="厳しい定義(置いて、離して、上げる)の結果で集計する(ある試行だけ)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = {}
    for t in args.tags:      # 判定し直した結果(rejudge_strict.py)があれば、それを使う
        name = args.results if os.path.exists(os.path.join(args.root, t, args.results)) else "results.jsonl"
        data[t] = load(args.root, t, args.strict, name)
    summary, md = {}, []
    for tag, rows in data.items():
        by = defaultdict(list)
        for (c, s), r in rows.items():
            by[c].append(r)
        conds = [c for c in CONDITIONS if c in by] + sorted(set(by) - set(CONDITIONS))
        summary[tag] = {}
        md += [f"### {tag}", "", "| 条件 | 本数 | 成功 | 成功率 [95% CI] | つかめた | 持ち上げた | ゴールまでの距離 中央値 (cm) | base と同じ配置での食い違い (base だけ / 条件だけ成功), p |",
               "|---|---|---|---|---|---|---|---|"]
        for c in conds:
            rs = sorted(by[c], key=lambda r: r["seed"])
            n, k = len(rs), sum(r["success"] for r in rs)
            st = [stage(r) for r in rs]
            lo, hi = wilson(k, n)
            row = {"n": n, "success": k, "rate": k / n, "ci95": [lo, hi],
                   "stages": {s: st.count(s) for s in STAGES},
                   "grasped": sum(s != "no_grasp" for s in st), "lifted": sum(s in ("lifted", "placed_off", "success") for s in st),
                   "dist_median_cm": float(np.median([r["dist_to_target"] for r in rs]) * 100),
                   "seeds": [r["seed"] for r in rs], "outcomes": [r["outcome"] for r in rs]}
            if any("reach_err_box" in r for r in rs):
                row["reach_err_box_cm"] = [round(r.get("reach_err_box", float("nan")) * 100, 2) for r in rs]
            paired = ""
            if c not in ("base", "pos_outer") and "base" in by:
                base = {r["seed"]: r["success"] for r in by["base"]}
                common = [r for r in rs if r["seed"] in base]
                b = sum(base[r["seed"]] and not r["success"] for r in common)
                cc = sum(r["success"] and not base[r["seed"]] for r in common)
                p = mcnemar_exact(b, cc)
                base_task = {r["seed"]: r.get("task", {}) for r in by["base"]}
                same = lambda r, k: r.get("task", {}).get(k) == base_task[r["seed"]].get(k)  # noqa: E731
                n_goal = sum(same(r, "target_xy") for r in common)
                n_ready = sum(same(r, "ready_tcp") for r in common)
                row["paired_vs_base"] = {"n_common": len(common), "base_only": b, "cond_only": cc, "p_mcnemar_exact": p,
                                         "same_goal_xy": n_goal, "same_ready_tcp": n_ready}
                paired = f"{b} / {cc} (n={len(common)}), p={p:.2f}; ゴール同じ {n_goal}、開始同じ {n_ready}"
            summary[tag][c] = row
            md.append(f"| {c} | {n} | {k} | {k / n:.0%} [{lo:.0%}, {hi:.0%}] | {row['grasped']} | {row['lifted']} | "
                      f"{row['dist_median_cm']:.1f} | {paired} |")
        md.append("")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # 図: 条件ごとの到達段階の割合(タグごとに横並びの小さな図)
    fig, axes = plt.subplots(1, len(data), figsize=(4.6 * len(data) + 1.6, 3.8), sharey=True, facecolor=SURFACE,
                             squeeze=False)
    for ax, (tag, s) in zip(axes[0], summary.items()):
        conds = list(s)
        y = np.arange(len(conds))[::-1]
        left = np.zeros(len(conds))
        for st in STAGES:
            w = np.array([s[c]["stages"][st] / s[c]["n"] for c in conds])
            ax.barh(y, w, left=left, height=0.6, color=STAGE_COLOR[st], edgecolor=SURFACE, linewidth=2,
                    label=STAGE_LABEL[st])
            left += w
        for yi, c in zip(y, conds):
            ax.text(1.02, yi, f"{s[c]['success']}/{s[c]['n']}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, conds)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_title(tag_label(tag), loc="left", fontsize=11, color=INK)
        ax.set_facecolor(SURFACE)
        for k in ("top", "right", "left"):
            ax.spines[k].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), frameon=False, fontsize=9, labelcolor=INK,
                       title="stage reached (right: successes)", title_fontsize=9)
    fig.tight_layout()
    path = os.path.join(args.out, "stages.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print("saved", path)


if __name__ == "__main__":
    main()
