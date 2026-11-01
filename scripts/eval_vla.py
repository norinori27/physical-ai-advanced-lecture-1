"""置き先タスク(task.py)で、openpi の推論サーバの方策を評価し、考察に必要な情報をすべて記録する。

DROID と同じく、画像は resize_with_pad で 224x224 にして送り、返ってきた行動チャンクの先頭 8 ステップを実行する(15 Hz)。
環境は教師(gen_demos.py)と同じ: DroidStyleEnv(retarget=True, track_yaw=True, markers=True)。

記録(エピソードごと):
  <out>/<tag>/<cond>_s<seed>.npz : 各ステップの観測(画像 2 枚は 180x320 の元画像、関節角、グリッパ)、実行した行動、
      問い合わせごとの行動チャンク全体(実行しなかった部分も)、シミュの正解(TCP、箱の位置と姿勢、指と箱の接触)
  <out>/<tag>/<cond>_s<seed>.mp4 : 外部 + 手首の動画
  <out>/<tag>/results.jsonl      : 成功、失敗の分類、タスクの全変数、所要ステップなど
失敗の分類: no_grasp(一度も持ち上がらない)/ dropped(運搬中に落とした)/ wrong_marker(違う色のマーカーに置いた)/
            misplaced(どのマーカーにも乗っていない)/ holding(時間切れで持ったまま)

使い方: MUJOCO_GL=osmesa python scripts/eval_vla.py --tag base --conditions base goal_color --seeds 5 [--host H --port P]
        --fake zero で配線だけ確認できる(サーバ不要)。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
EVAL_SEED0 = 500_000      # 学習(0〜)、検証(900000〜)と重ならない


def pad224(img):
    from openpi_client import image_tools
    return image_tools.resize_with_pad(img, 224, 224)


def classify(env, task, lifted, events):
    from openarm_anomaly.task import MARKER_HALF, place_success
    if place_success(env, task):
        return "success"
    box = env.box_pos()
    if not lifted:
        return "no_grasp"
    if box[2] > env.box_half + 0.01:
        return "holding"
    for c, (x, y) in task.markers:
        if abs(box[0] - x) < MARKER_HALF and abs(box[1] - y) < MARKER_HALF and c != task.target:
            return "wrong_marker"
    return "dropped" if events.get("dropped_far") else "misplaced"


def run(policy_infer, task, max_steps, horizon, record_images=True, post_steps=0):
    """post_steps > 0: 成功と判定した後も、その本数だけ方策を動かし続ける(放した後に手を上げるところまで撮るため)。
    結果(outcome、dist_to_target など)は、これまでと同じく成功と判定した時点の値。後の様子は summary["post"] に入れる。"""
    import mujoco
    from openarm_anomaly.task import apply_task, place_success
    from openarm_anomaly.vla_env import DroidStyleEnv

    env = DroidStyleEnv(box_xy=task.box_xy, retarget=True, ready_tcp=task.ready_tcp, track_yaw=True, markers=True)
    apply_task(env, task)
    m, d = env.m, env.d
    # 箱とゴールの真上で、真下向きのハンドが届くか(位置の誤差 m)。外側の条件では腕の届く端になり、届かないことがある
    reach = {}
    for name, xy in (("box", task.box_xy), ("goal", task.target_xy)):
        _, err = env.kin.solve_global(np.array([*xy, env.box_half]), n_starts=10)
        reach[name] = float(err)
    box_gid = m.geom("box_geom").id
    pads = {m.geom("openarm_left_finger_pad").id, m.geom("openarm_right_finger_pad").id}
    rec = {k: [] for k in ["ext", "wrist", "joint_position", "gripper_position", "action", "tcp", "box_pos", "box_quat",
                           "pad_contact", "q_openarm"]}
    chunks, chunk_t, infer_ms = [], [], []
    pending = np.zeros((0, 8))
    lifted, events, success_run = False, {}, 0
    at_success, end_t = None, max_steps
    obs = env.obs(task.prompt)
    for t in range(max_steps):
        if t >= end_t:
            break
        if len(pending) == 0:
            req = dict(obs)
            req["observation/exterior_image_1_left"] = pad224(obs["observation/exterior_image_1_left"])
            req["observation/wrist_image_left"] = pad224(obs["observation/wrist_image_left"])
            t0 = time.time()
            chunk = np.asarray(policy_infer(req))
            infer_ms.append((time.time() - t0) * 1000)
            chunks.append(chunk)
            chunk_t.append(t)
            pending = chunk[:horizon]
        a = np.asarray(pending[0], float)
        pending = pending[1:]
        if record_images:
            rec["ext"].append(obs["observation/exterior_image_1_left"])
            rec["wrist"].append(obs["observation/wrist_image_left"])
        rec["joint_position"].append(obs["observation/joint_position"])
        rec["gripper_position"].append(obs["observation/gripper_position"])
        rec["action"].append(a)
        env.step(a)
        jb = m.joint("box_free").qposadr[0]
        rec["tcp"].append(env.kin.fk(d.qpos[:7])[0])
        rec["box_pos"].append(d.qpos[jb:jb + 3].copy())
        rec["box_quat"].append(d.qpos[jb + 3:jb + 7].copy())
        rec["q_openarm"].append(d.qpos[:7].copy())
        rec["pad_contact"].append(sum(1 for c in d.contact[:d.ncon]
                                      if {c.geom1, c.geom2} & pads and box_gid in (c.geom1, c.geom2)))
        if d.qpos[jb + 2] > env.box_half + 0.04:
            lifted = True
        if lifted and d.qpos[jb + 2] < env.box_half + 0.01 and env.gripper_position() > 0.5:
            events.setdefault("dropped_far", True)
        obs = env.obs(task.prompt)
        success_run = success_run + 1 if place_success(env, task) else 0
        if at_success is None and success_run >= 8:     # 置いて放した状態が 8 ステップ続いたら成功(ここで判定する)
            at_success = {"t": t + 1, "box": env.box_pos().copy(), "outcome": classify(env, task, lifted, events)}
            end_t = t + 1 + post_steps
    post = None
    if at_success is not None:
        outcome, box = at_success["outcome"], at_success["box"]
        if post_steps > 0:      # 成功の後の様子: 箱が動いていないか、手が離れたか
            box_end = env.box_pos()
            tcp_end = env.kin.fk(d.qpos[:7])[0]
            post = {"steps": int(len(rec["action"]) - at_success["t"]), "still_on_target": bool(place_success(env, task)),
                    "box_moved_cm": float(np.linalg.norm(box_end - box) * 100),
                    "tcp_above_box_cm": float((tcp_end[2] - box_end[2]) * 100),
                    "pad_contact_end": int(rec["pad_contact"][-1]), "gripper_end": float(env.gripper_position())}
        t = at_success["t"] - 1
    else:
        outcome = classify(env, task, lifted, events)
        box = env.box_pos()
    env.close()
    rec = {k: np.asarray(v) for k, v in rec.items() if len(v)}
    rec["chunks"], rec["chunk_t"] = np.asarray(chunks), np.asarray(chunk_t)
    summary = {"outcome": outcome, "success": outcome == "success", "steps": t + 1, "lifted": lifted,
               "box_final": box.tolist(), "dist_to_target": float(np.hypot(box[0] - task.target_xy[0], box[1] - task.target_xy[1])),
               "infer_ms_mean": float(np.mean(infer_ms)), "n_infer": len(chunks),
               "reach_err_box": reach["box"], "reach_err_goal": reach["goal"], "n_frames": len(rec["action"])}
    if post is not None:
        summary["post"] = post
    return summary, rec


def main():
    from openarm_anomaly.task import CONDITIONS, sample_task

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tag", required=True, help="結果のフォルダ名(base、run1_s3000 など)")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--post-steps", type=int, default=0, help="成功の後も動かし続けるステップ数(動画用。判定は成功時点)")
    ap.add_argument("--only", nargs="*", default=None, help="この試行だけ(例: base:500001 goal_color:500004)")
    ap.add_argument("--fake", choices=["zero"])
    args = ap.parse_args()

    if args.fake:
        policy_infer = lambda req: np.zeros((16, 8))  # noqa: E731
        globals()["pad224"] = lambda img: img
    else:
        from openpi_client import websocket_client_policy
        client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        policy_infer = lambda req: client.infer(req)["actions"]  # noqa: E731

    import mediapy
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    only = {(x.split(":")[0], int(x.split(":")[1])) for x in args.only} if args.only else None
    res_path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(res_path):     # 途中で止まったときは、済んだエピソードを飛ばして続きから
        with open(res_path) as f:
            done = {(r["condition"], r["seed"]) for r in (json.loads(x) for x in f if x.strip())}
    for cond in args.conditions:
        for i in range(args.seeds):
            seed = args.seed0 + i     # 条件ごとに同じ seed 列(条件間で配置の乱数を揃える)
            if only is not None and (cond, seed) not in only:
                continue
            if (cond, seed) in done:
                print(f"{cond}_s{seed}: done, skip", flush=True)
                continue
            task = sample_task(np.random.default_rng(seed), cond, seed)
            summary, rec = run(policy_infer, task, args.max_steps, args.horizon, post_steps=args.post_steps)
            name = f"{cond}_s{seed}"
            np.savez_compressed(os.path.join(out_dir, name + ".npz"), task_json=json.dumps(task.to_dict()), **rec)
            mediapy.write_video(os.path.join(out_dir, name + ".mp4"), np.concatenate([rec["ext"], rec["wrist"]], 2), fps=15)
            line = {"tag": args.tag, "condition": cond, "seed": seed, "task": task.to_dict(), **summary}
            with open(res_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"{name}: {summary['outcome']} steps {summary['steps']} dist {summary['dist_to_target']:.3f} "
                  f"prompt '{task.prompt}'", flush=True)


if __name__ == "__main__":
    main()
