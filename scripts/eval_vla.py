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


def run(policy_infer, task, max_steps, horizon, record_images=True):
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
    obs = env.obs(task.prompt)
    for t in range(max_steps):
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
        if success_run >= 8:          # 置いて放した状態が 8 ステップ続いたら終了
            break
    outcome = classify(env, task, lifted, events)
    box = env.box_pos()
    env.close()
    rec = {k: np.asarray(v) for k, v in rec.items() if len(v)}
    rec["chunks"], rec["chunk_t"] = np.asarray(chunks), np.asarray(chunk_t)
    summary = {"outcome": outcome, "success": outcome == "success", "steps": t + 1, "lifted": lifted,
               "box_final": box.tolist(), "dist_to_target": float(np.hypot(box[0] - task.target_xy[0], box[1] - task.target_xy[1])),
               "infer_ms_mean": float(np.mean(infer_ms)), "n_infer": len(chunks),
               "reach_err_box": reach["box"], "reach_err_goal": reach["goal"]}
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
    res_path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(res_path):     # 途中で止まったときは、済んだエピソードを飛ばして続きから
        with open(res_path) as f:
            done = {(r["condition"], r["seed"]) for r in (json.loads(x) for x in f if x.strip())}
    for cond in args.conditions:
        for i in range(args.seeds):
            seed = args.seed0 + i     # 条件ごとに同じ seed 列(条件間で配置の乱数を揃える)
            if (cond, seed) in done:
                print(f"{cond}_s{seed}: done, skip", flush=True)
                continue
            task = sample_task(np.random.default_rng(seed), cond, seed)
            summary, rec = run(policy_infer, task, args.max_steps, args.horizon)
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
