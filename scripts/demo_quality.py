"""デモの途中の質を調べる(成功したエピソードだけを保存しているが、途中の動きがきれいとは限らない)。

エピソードごとに:
  stall_frames : 手先(仮想 Franka の順運動学)がほぼ動かず(1 ステップ 1 mm 未満)、グリッパの指令も変わらないフレーム数
  max_jump_cm  : 1 ステップの手先の移動の最大(教師の上限は 2 cm。大きければ腕が振れた)
  clip_frac    : 腕の行動が ±1 に張り付いた割合
  maxed_phases : 上限のステップ数まで使い切った段階(経由点に収束しなかった)
使い方: python scripts/demo_quality.py --src data/demos [--out outputs/demo_check/demo_quality.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from openarm_anomaly.franka_retarget import franka_fk  # noqa: E402

PHASE_MAX = {"approach": 60, "align": 40, "descend": 40, "lift": 40, "carry": 80, "lower": 50,
             "release": 20, "retreat_slow": 15, "retreat": 30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/demos")
    ap.add_argument("--out", default="outputs/demo_check/demo_quality.json")
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(args.src, "ep_*.npz"))):
        with np.load(f) as d:
            q, a, ph = d["joint_position"], d["actions"], list(d["phase"])
        p = np.array([franka_fk(x)[0] for x in q])
        step = np.r_[np.linalg.norm(np.diff(p, axis=0), axis=1), 0]
        grip_change = np.r_[np.abs(np.diff(a[:, 7])) > 0.5, False]
        stall = (step < 0.001) & ~grip_change
        stall_by_phase = {k: int(stall[[i for i, x in enumerate(ph) if x == k]].sum()) for k in dict.fromkeys(ph)}
        counts = {k: ph.count(k) for k in dict.fromkeys(ph)}
        rows.append({"file": os.path.basename(f), "steps": len(a), "stall_frames": int(stall.sum()),
                     "stall_by_phase": stall_by_phase, "max_jump_cm": float(step.max() * 100),
                     "clip_frac": float(np.mean(np.abs(a[:, :7]) >= 0.999)),
                     "maxed_phases": [k for k, n in counts.items() if n >= PHASE_MAX.get(k, 1e9)]})
    st = np.array([r["stall_frames"] for r in rows])
    jumps = np.array([r["max_jump_cm"] for r in rows])
    total = sum(r["steps"] for r in rows)
    print(f"{len(rows)} 本、{total} フレーム")
    print(f"止まっているフレーム: 合計 {st.sum()}({st.sum() / total:.1%})、1 本あたり 中央値 {np.median(st):.0f} 最大 {st.max()}")
    by = {}
    for r in rows:
        for k, v in r["stall_by_phase"].items():
            by[k] = by.get(k, 0) + v
    print("  段階別:", dict(sorted(by.items(), key=lambda x: -x[1])))
    print(f"1 ステップの手先の移動の最大: 中央値 {np.median(jumps):.1f} cm、>3 cm の本数 {np.sum(jumps > 3)}、>5 cm {np.sum(jumps > 5)}")
    print(f"行動が ±1 に張り付いた割合: 平均 {np.mean([r['clip_frac'] for r in rows]):.2%}")
    mp = {}
    for r in rows:
        for k in r["maxed_phases"]:
            mp[k] = mp.get(k, 0) + 1
    print("上限まで使い切った段階(本数):", mp)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
