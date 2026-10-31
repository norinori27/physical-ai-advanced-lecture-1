"""学習用デモを、水準の組み合わせ(箱の色 × ゴールの色 × 言い回し = 12 通り)ごとに同数にそろえる。

多い組み合わせは seed の小さい順に --per 本だけ残し、残りは <src>_extra/ に移す(消さない)。
足りない組み合わせは、gen_demos.py --force で追加生成するコマンドを表示する(--run なら実行する)。

使い方: python scripts/balance_demos.py --src data/demos --per 20 [--run]
"""
import argparse
import glob
import itertools
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, "src")
from openarm_anomaly.task import TEMPLATES_TRAIN  # noqa: E402


def combos_in(src):
    out = {}
    for f in sorted(glob.glob(os.path.join(src, "ep_*.npz"))):
        with np.load(f) as d:
            t = json.loads(str(d["task_json"]))
        key = (t["box_color"], t["target"], TEMPLATES_TRAIN.index(t["template"]))
        out.setdefault(key, []).append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/demos")
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    extra = args.src.rstrip("/") + "_extra"
    os.makedirs(extra, exist_ok=True)
    have = combos_in(args.src)
    levels = list(itertools.product(("orange", "red"), ("blue", "green"), range(len(TEMPLATES_TRAIN))))
    cmds = []
    for k, key in enumerate(levels):
        files = have.get(key, [])
        for f in files[args.per:]:
            shutil.move(f, os.path.join(extra, os.path.basename(f)))
        need = args.per - min(len(files), args.per)
        print(f"{key}: {len(files)} 本 → 残す {min(len(files), args.per)}、移す {max(len(files) - args.per, 0)}、追加 {need}")
        if need:
            force = json.dumps({"box_color": key[0], "target": key[1], "template": key[2]})
            cmds.append([sys.executable, "-u", "scripts/gen_demos.py", "--n", str(need), "--workers", str(args.workers),
                         "--seed0", str(100_000 + 10_000 * k), "--out", args.src, "--force", force])
    for c in cmds:
        print(" ".join(c))
        if args.run:
            subprocess.run(c, check=True, env={**os.environ, "MUJOCO_GL": "osmesa"})
    if args.run:   # 追加生成でも最後の 1 回分の超過が出るので、もう一度そろえる
        have = combos_in(args.src)
        for key, files in have.items():
            for f in files[args.per:]:
                shutil.move(f, os.path.join(extra, os.path.basename(f)))
        final = {k: len(v[:args.per]) for k, v in combos_in(args.src).items()}
        print("最終:", final, "合計", sum(final.values()))


if __name__ == "__main__":
    main()
