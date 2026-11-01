"""評価と同じ配置(条件・seed)で教師(スクリプト制御)を動かし、評価と同じ形式で動画と結果を保存する(動画の比較用)。

配置は eval_vla.py と同じく sample_task(default_rng(seed), condition, seed) で作る。教師は箱とマーカーの正確な位置を使う。
既定では DART の雑音なし(--dart-sigma 0)。デモを作ったときは σ=0.05 だった。
出力: <out>/<条件>_s<seed>.mp4(外部 | 手首)、<out>/results.jsonl
使い方: MUJOCO_GL=osmesa python scripts/teacher_eval.py [--conditions base goal_color] [--seeds 10] [--workers 6]
"""
import argparse
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

EVAL_SEED0 = 500000


def run(job):
    cond, seed, sigma, out, max_steps = job
    import mediapy
    from gen_demos import run_episode
    from rejudge_strict import judge
    info, data = run_episode(seed, render=True, dart_sigma=sigma, condition=cond, max_steps=max_steps)
    frames = np.concatenate([data["exterior_image_1_left"], data["wrist_image_left"]], 2)
    mediapy.write_video(os.path.join(out, f"{cond}_s{seed}.mp4"), frames, fps=15)
    t = info["task"]
    box = np.asarray(info["box_final"])
    lifted = info["phase_box"].get("lift", [0, 0, 0])[2] > 0.035 + 0.04
    # 学習後のモデルと同じ厳しい定義(rejudge_strict.py: 置いて 8 ステップ、離して 5 cm 以上上げたのが 1 ステップ)
    j = judge(data, release_cm=5.0, hold=1)
    s_out = "success" if j["t_release"] is not None else ("placed_not_released" if j["t_place"] is not None else None)
    return {"tag": "teacher", "condition": cond, "seed": seed, "task": t, "success": bool(info["success"]),
            "outcome": "success" if info["success"] else ("misplaced" if lifted else "no_grasp"), "lifted": bool(lifted),
            "steps": info["steps"], "box_final": box.tolist(),
            "dist_to_target": float(np.hypot(box[0] - t["target_xy"][0], box[1] - t["target_xy"][1])),
            "phase_steps": info["phase_steps"], "dart_sigma": sigma,
            "strict": {"release_cm": 5.0, "hold": 1, "success": s_out == "success",
                       "outcome": s_out or ("misplaced" if lifted else "no_grasp"),
                       "t": None if j["t_release"] is None else j["t_release"] + 1, "dist_to_target": j["dist"],
                       "t_place": None if j["t_place"] is None else j["t_place"] + 1, "end": j}}


def main():
    from openarm_anomaly.task import CONDITIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--dart-sigma", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--out", default="outputs/eval/teacher_strict")
    ap.add_argument("--workers", type=int, default=6)     # 描画も CPU を使うので、合計 8 以下(CLAUDE.md)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        with open(res_path) as f:
            done = {(r["condition"], r["seed"]) for r in (json.loads(x) for x in f if x.strip())}
    jobs = [(c, EVAL_SEED0 + i, args.dart_sigma, args.out, args.max_steps)
            for c in args.conditions for i in range(args.seeds) if (c, EVAL_SEED0 + i) not in done]
    with get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(run, jobs):
            with open(res_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{r['condition']}_s{r['seed']}: {r['outcome']} steps {r['steps']} dist {r['dist_to_target']:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
