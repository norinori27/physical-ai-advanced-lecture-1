"""DAgger(切り替え型): モデルを動かした記録から、モデルがずれた時点の状態を作り直し、そこから教師(scripts/dagger.py)で動かした
軌道を学習データにする。置いて、離して、手を上げられた軌道で、切り替えの直後に腕や箱が跳ねていないものだけを残す。

切り替える時点: 箱を持っていて、ゴールから水平に 3〜15 cm ずれている時点(運んでいる途中と、ずれたまま降ろし始めた時点)。
  1 本の記録から、8 ステップ以上離れた時点を最大 --per-episode 個試す(降り始めた時点を優先)。
状態の作り直し: dagger_takeover.py と同じ(腕・箱は記録の位置、指は握り直させる、1 ステップの関節の変化に上限)。
  評価のときの IK の出発点は記録に無いので、ふだんと違う姿勢では作り直した直後に跳ねることがある → 跳ねたものは捨てる。
出力: gen_demos.py と同じ形の ep_<seed><試行番号>.npz(画像、仮想 Franka の関節角、グリッパ、教師の行動、指示文)
使い方: MUJOCO_GL=osmesa python scripts/dagger_collect.py --src outputs/eval_npz/dagger1_p0/dagger1_p0 --out data/demos_dagger1
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

MAX_STEPS = 250
JUMP_RAD = 0.25       # 切り替え後 5 ステップの間に、1 ステップで OpenArm の関節がこれ以上動いたら跳ねたとみなす
BOX_JUMP = 0.08       # 1 ステップで箱がこれ以上動いたら跳ねたとみなす(m)。教師の速い移動は 1 ステップ 2 cm 以下


def candidates(z, h, per_episode):
    task = json.loads(str(z["task_json"]))
    tg = np.asarray(task["target_xy"])
    box, pad, grip = z["box_pos"], z["pad_contact"], np.ravel(z["gripper_position"])
    low, high = [], []
    for t in range(2, len(box)):
        e = np.linalg.norm(box[t - 1, :2] - tg)
        if pad[t - 1] > 0 and grip[t] > 0.1 and 0.03 < e < 0.15 and box[t - 1, 2] > h + 0.01:
            (low if box[t - 1, 2] < h + 0.06 else high).append(t)
    out = []
    for t in low + high:          # 降り始めた時点を優先
        if all(abs(t - u) >= 8 for u in out):
            out.append(t)
        if len(out) >= per_episode:
            break
    return out


def attempt(path, t0):
    import mujoco
    from dagger import Context, env_state, teacher
    from openarm_anomaly.scripted import GRIP_CLOSE, GRIP_OPEN, N_ARM
    from openarm_anomaly.task import apply_task, place_success, sample_task
    from openarm_anomaly.vla_env import DroidStyleEnv
    z = np.load(path)
    task_d = json.loads(str(z["task_json"]))
    seed = int(os.path.basename(path).rsplit("_s", 1)[1][:-4])
    task = sample_task(np.random.default_rng(seed), task_d["condition"], seed)
    env = DroidStyleEnv(box_xy=task.box_xy, retarget=True, ready_tcp=task.ready_tcp, track_yaw=True, markers=True)
    apply_task(env, task)
    m, d = env.m, env.d
    ctx = Context(env.kin, task.to_dict(), z["q_openarm"][0], z["joint_position"][0], env.rt.offset, env.box_half)
    jb, vb = m.joint("box_free").qposadr[0], m.joint("box_free").dofadr[0]
    q = z["q_openarm"][t0 - 1]
    finger = GRIP_OPEN - float(np.ravel(z["gripper_position"])[t0]) * (GRIP_OPEN - GRIP_CLOSE)
    d.qpos[N_ARM:N_ARM + 2] = finger + 0.004
    for _ in range(80):
        d.qpos[:N_ARM] = q
        d.qpos[jb:jb + 3] = z["box_pos"][t0 - 1]
        d.qpos[jb + 3:jb + 7] = z["box_quat"][t0 - 1]
        d.qvel[:N_ARM] = 0
        d.qvel[vb:vb + 6] = 0
        d.ctrl[:N_ARM] = q
        d.ctrl[N_ARM] = GRIP_CLOSE
        mujoco.mj_step(m, d)
    d.qpos[:N_ARM] = q
    d.qpos[jb:jb + 3] = z["box_pos"][t0 - 1]
    d.qpos[jb + 3:jb + 7] = z["box_quat"][t0 - 1]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    env.q_cmd = q.copy()
    env.max_dq = 0.15
    env.rt.q = z["joint_position"][t0].astype(float).copy()
    env.rt.q_cmd = env.rt.q.copy()
    pads = {m.geom("openarm_left_finger_pad").id, m.geom("openarm_right_finger_pad").id}
    box_gid = m.geom("box_geom").id
    rec = {k: [] for k in ("ext", "wrist", "q_f", "grip", "act", "phase")}
    success_t, jump, prev_q, prev_box = None, False, d.qpos[:N_ARM].copy(), env.box_pos()
    for k in range(MAX_STEPS):
        obs = env.obs(task.prompt)
        s = env_state(env, pads, box_gid)
        a, name = teacher(ctx, env.kin, s)
        rec["ext"].append(obs["observation/exterior_image_1_left"])
        rec["wrist"].append(obs["observation/wrist_image_left"])
        rec["q_f"].append(obs["observation/joint_position"].astype(np.float32))
        rec["grip"].append(np.float32(obs["observation/gripper_position"][0]))
        rec["act"].append(a.astype(np.float32))
        rec["phase"].append(name)
        env.step(a)
        q_now, box_now = d.qpos[:N_ARM].copy(), env.box_pos()
        dq, db = np.abs(q_now - prev_q).max(), np.linalg.norm(box_now - prev_box)
        if (k < 5 and dq > JUMP_RAD) or db > BOX_JUMP:
            jump = f"k={k} dq={dq:.2f} dbox={db:.3f}"
            break
        prev_q, prev_box = q_now, box_now
        s2 = env_state(env, pads, box_gid)
        if success_t is None and place_success(env, task) and s2["pad"] == 0 and s2["tcp"][2] - s2["box"][2] > 0.05:
            success_t = k
        if success_t is not None and k >= success_t + 3:     # 教師のデモと同じく、手を上げたら少しで終える
            break
    box = env.box_pos()
    env.close()
    info = {"file": os.path.basename(path), "t0": t0, "success": success_t is not None and not jump, "jump": jump,
            "steps": len(rec["act"]), "dist_cm": round(float(np.hypot(*(box[:2] - np.asarray(task.target_xy)))) * 100, 1),
            "err0_cm": round(float(np.linalg.norm(z["box_pos"][t0 - 1, :2] - np.asarray(task.target_xy))) * 100, 1)}
    return info, rec, task_d


def work(job):
    path, out, per_episode = job
    from openarm_anomaly.vla_env import DroidStyleEnv  # noqa: F401
    z = np.load(path)
    res = []
    seed = int(os.path.basename(path).rsplit("_s", 1)[1][:-4])
    for i, t0 in enumerate(candidates(z, 0.035, per_episode)):
        info, rec, task_d = attempt(path, t0)
        if info["success"]:
            np.savez_compressed(os.path.join(out, f"ep_{seed:06d}{i}.npz"),
                                exterior_image_1_left=np.array(rec["ext"], np.uint8), wrist_image_left=np.array(rec["wrist"], np.uint8),
                                joint_position=np.array(rec["q_f"]), gripper_position=np.array(rec["grip"])[:, None],
                                actions=np.array(rec["act"]), prompt=np.array(task_d["prompt"]), phase=np.array(rec["phase"]),
                                task_json=np.array(json.dumps(task_d)), takeover_t=np.array(t0))
        res.append(info)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", default="data/demos_dagger1")
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from openarm_anomaly.vla_env import DroidStyleEnv
    DroidStyleEnv(markers=True).close()     # 描画用の間引いたメッシュのキャッシュを、並列処理の前に 1 回だけ作る
    files = sorted(f for d in args.src for f in glob.glob(os.path.join(d, "*.npz")))
    allres = []
    with get_context("spawn").Pool(args.workers) as pool:
        for res in pool.imap_unordered(work, [(f, args.out, args.per_episode) for f in files]):
            for r in res:
                print(r, flush=True)
            allres += res
    n_ep = len({r["file"] for r in allres})
    print(f"episodes with takeover points {n_ep}/{len(files)}, attempts {len(allres)}, "
          f"kept {sum(r['success'] for r in allres)}, jumped {sum(bool(r['jump']) for r in allres)}")
    with open(os.path.join(args.out, "collect_summary.json"), "w") as f:
        json.dump(allres, f, indent=1)


if __name__ == "__main__":
    main()
