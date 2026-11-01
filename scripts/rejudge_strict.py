"""評価の記録(npz)から、厳しい定義(置いて、離して、上げる)を判定し直す。評価を撮り直さずに定義の細部を変えるため。

  置いた   : place_success と同じ(箱がゴールのマーカーの上、机に置かれている、グリッパの開き < 0.5)が 8 ステップ続く
             注意: 箱をつかんだままでも開きは約 0.18 なので、これだけでは「離した」ことにならない
  離して上げた: 置いた後に、箱がゴールの上にあるまま、指と箱が接しておらず、手先が箱の中心より release_cm 以上高い状態が
             hold ステップ続く(既定 1。教師のデモは上げてすぐ終わるので、8 ステップ続くことを求めると教師より厳しい)
  最後のフレームで箱がゴールの上にあっても、8 ステップ続かなかった・離していなければ成功にしない
出力: <root>/<tag>/results_rejudged.jsonl(results.jsonl の "strict" を置き換えたもの。summarize_eval.py --results で使う)
使い方: python scripts/rejudge_strict.py --tag run1_s14999_strict [--npz outputs/eval_npz/run1_s14999_strict/run1_s14999_strict]
"""
import argparse
import json
import os

import numpy as np

BOX_HALF, MARKER_HALF, PLACE_HOLD = 0.035, 0.045, 8


def judge(z, release_cm, hold):
    task = json.loads(str(z["task_json"]))
    tx, ty = task["target_xy"]
    box, tcp = z["box_pos"], z["tcp"]
    grip, pad = np.ravel(z["gripper_position"]), z["pad_contact"]
    on_target = (np.abs(box[:, 0] - tx) < MARKER_HALF) & (np.abs(box[:, 1] - ty) < MARKER_HALF)
    placed = on_target & (box[:, 2] < BOX_HALF + 0.01) & (grip < 0.5)
    released = placed & (pad == 0) & (tcp[:, 2] - box[:, 2] >= release_cm / 100)
    t_place, t_rel, run = None, None, 0
    for t in range(len(placed)):
        run = run + 1 if placed[t] else 0
        if t_place is None and run >= PLACE_HOLD:
            t_place = t
    if t_place is not None:
        run = 0
        for t in range(t_place - PLACE_HOLD + 1, len(released)):
            run = run + 1 if released[t] else 0
            if run >= hold:
                t_rel = t
                break
    dist = lambda i: float(np.hypot(box[i, 0] - tx, box[i, 1] - ty))
    return {"t_place": t_place, "t_release": t_rel, "dist": dist(t_rel if t_rel is not None else -1),
            "on_target_end": bool(on_target[-1]), "pad_contact_end": int(pad[-1]),
            "tcp_above_box_end_cm": float((tcp[-1, 2] - box[-1, 2]) * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default="outputs/eval")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--release-cm", type=float, default=5.0)
    ap.add_argument("--hold", type=int, default=1)
    args = ap.parse_args()
    npz_dir = args.npz or f"outputs/eval_npz/{args.tag}/{args.tag}"
    out = []
    for line in open(os.path.join(args.root, args.tag, "results.jsonl")):
        if not line.strip():
            continue
        r = json.loads(line)
        path = os.path.join(npz_dir, f"{r['condition']}_s{r['seed']}.npz")
        if not os.path.exists(path):
            print(f"no npz: {path}")
            continue
        j = judge(np.load(path, allow_pickle=True), args.release_cm, args.hold)
        old = r.get("strict", {}).get("outcome")
        if j["t_release"] is not None:
            o = "success"
        elif j["t_place"] is not None:
            o = "placed_not_released"
        elif r["outcome"] == "success":      # 最後のフレームだけゴールの上(つかんだまま、または 8 ステップ続かない)
            o = "holding" if j["pad_contact_end"] else "misplaced"
        else:
            o = r["outcome"]
        r["strict"] = {"release_cm": args.release_cm, "hold": args.hold, "success": o == "success", "outcome": o,
                       "t": None if j["t_release"] is None else j["t_release"] + 1, "dist_to_target": j["dist"],
                       "t_place": None if j["t_place"] is None else j["t_place"] + 1,
                       "end": {k: j[k] for k in ("on_target_end", "pad_contact_end", "tcp_above_box_end_cm")}}
        out.append(r)
        print(f"{r['condition']}_s{r['seed']}: {o:20s} (eval: {r['outcome']} / strict8: {old}) "
              f"place t={r['strict']['t_place']} release t={r['strict']['t']} "
              f"end: above {j['tcp_above_box_end_cm']:.1f} cm, pad {j['pad_contact_end']}")
    path = os.path.join(args.root, args.tag, "results_rejudged.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n = sum(r["strict"]["success"] for r in out)
    print(f"saved {path}: {n}/{len(out)} success (release >= {args.release_cm} cm, hold {args.hold})")


if __name__ == "__main__":
    main()
