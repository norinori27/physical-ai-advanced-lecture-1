"""微調整用のデモ(箱を指定した色のマーカーに置く)を、仮想 Franka の関節空間(pi05_droid と同じ入出力)で作る。

教師はシミュの正解(箱とゴールの位置)を使い、仮想 Franka の手先を
  箱の上 → 箱の 4 cm 上で合わせ直す → ゆっくり降下 → 把持 → 持ち上げ → ゴールの上 → ゆっくり降ろす → 放す → 上げる
と動かす。各ステップで、実際の状態から次の経由点を Franka の IK で関節角にし、(目標 − 現在) / 0.2 を行動のラベルにする。
DART(Laskey et al., CoRL 2017): 実行する行動にはガウス雑音を足し、記録するラベルは雑音なしの教師の行動にする。
行動は評価と同じ経路(DroidStyleEnv(retarget=True, track_yaw=True))で OpenArm を動かす。ハンドのヨーは仮想 Franka の
手首で制御する(ヨーを自由にすると、ハンドの向きが IK の経路しだいになり、箱と合わずに把持を失敗した)。
シーンの変数は src/openarm_anomaly/task.py(condition="train")。置けたエピソードだけを保存する。

使い方(リポジトリのルートから):
    MUJOCO_GL=osmesa python scripts/gen_demos.py --n 240 --workers 8 --out data/demos           # 学習用
    MUJOCO_GL=osmesa python scripts/gen_demos.py --n 24 --workers 8 --seed0 900000 --out data/demos_val   # 検証用
    MUJOCO_GL=osmesa python scripts/gen_demos.py --n 16 --workers 8 --check                     # 保存せず成功率だけ
出力: <out>/ep_<seed>.npz(画像 2 種、関節角、グリッパ位置、行動、指示文、タスクの JSON)と <out>/summary.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")

DELTA = 0.2
DART_SIGMA = 0.05        # 実行時に関節速度へ足す雑音の標準偏差(行動の単位。0.05 = 1 ステップ 0.01 rad)
PRECISE_SCALE = 0.3      # 位置合わせが要る段階では雑音をこの倍率に弱める(0.1 では経由点に収束しなかった)
PRECISE = ("align", "descend", "grasp", "lower", "release", "retreat_slow")
LIFT = 0.15              # 持ち上げ・運搬の高さ(箱の中心の上、m)


def run_episode(seed, render=True, dart_sigma=DART_SIGMA, condition="train", task=None, max_steps=400, force=None):
    from openarm_anomaly.franka_retarget import franka_fk, franka_ik
    from openarm_anomaly.task import apply_task, place_success, sample_task
    from openarm_anomaly.vla_env import DroidStyleEnv

    rng = np.random.default_rng(seed)
    task = task or sample_task(rng, condition, seed, force=force)
    env = DroidStyleEnv(box_xy=task.box_xy, retarget=True, ready_tcp=task.ready_tcp, track_yaw=True, markers=True)
    box_yaw = apply_task(env, task)
    rt = env.rt
    _, R0 = franka_fk(rt.q)
    h = env.box_half
    # ハンドのヨー(OpenArm の TCP の x 軸の向き)。モデルと同じく、仮想 Franka の手首の回転で制御する。
    # 把持: 箱の向き。運搬以降: ゴールの位置で無理なくとれる向き。
    hand_yaw = lambda: float(np.arctan2(*env.kin.fk(env.d.qpos[:7])[1][[1, 0], 0]))
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    psi0 = hand_yaw()
    # 箱のヨーは「その位置で無理なくとれるハンドの向き ± 7°」に置いてある(apply_task)ので、それをそのまま狙う。
    # 90° ずらした等価な向きは、手首が回りきらず届かないことがある。
    yaw_grasp = psi0 + wrap(box_yaw - psi0)
    q_g = env.kin.solve(env.d.qpos[:7].copy(), np.array([*task.box_xy, h + LIFT]), iters=100)
    q_g = env.kin.solve(q_g, np.array([*task.target_xy, h + LIFT]), iters=100)
    q_g = env.kin.solve(q_g, np.array([*task.target_xy, h]), iters=100)
    _, R_goal = env.kin.fk(q_g)
    yaw_goal_nat = float(np.arctan2(R_goal[1, 0], R_goal[0, 0]))
    yaw_goal = yaw_grasp + wrap(yaw_goal_nat - yaw_grasp)
    box_f = np.array([*task.box_xy, h]) - rt.offset            # 箱の中心(Franka 座標)
    goal_f = np.array([*task.target_xy, h + 0.005]) - rt.offset  # 置くときの箱の中心
    hover = rng.uniform(0.10, 0.16)
    fast = rng.uniform(0.012, 0.02)                              # 1 ステップの手先の移動上限(m)、15 Hz
    slow = 0.006
    # (名前, 目標の手先, グリッパ, 最低ステップ数, 最大ステップ数, 1 ステップの移動上限)。最低ステップ後、目標に着いたら次へ。
    plan = [("approach", box_f + [0, 0, hover], 0.0, 0, 60, fast, yaw_grasp),
            ("align", box_f + [0, 0, 0.04], 0.0, 4, 40, fast, yaw_grasp),
            ("descend", box_f, 0.0, 0, 40, slow, yaw_grasp),
            ("grasp", box_f, 1.0, 8, 8, slow, yaw_grasp),
            ("lift", box_f + [0, 0, LIFT], 1.0, 0, 40, fast, yaw_grasp),
            ("carry", goal_f + [0, 0, LIFT], 1.0, 0, 80, fast, yaw_goal),
            ("lower", goal_f, 1.0, 0, 50, slow, yaw_goal),
            ("release", goal_f, 0.0, 6, 20, slow, yaw_goal),
            ("retreat_slow", goal_f + [0, 0, 0.04], 0.0, 0, 15, slow, yaw_goal),
            ("retreat", goal_f + [0, 0, 0.10], 0.0, 0, 30, fast, yaw_goal)]
    YAW_RATE = np.deg2rad(4)     # 1 ステップのヨーの変化の上限

    ext, wrist, qs, grips, acts, phases = [], [], [], [], [], []
    phase_box = {}       # 各段階の終わりの箱の位置(失敗の分類用)
    obs = env.obs(task.prompt) if render else None
    for name, p_goal, g, n_min, n_max, lim, yaw_t in plan:
        sigma = dart_sigma * (PRECISE_SCALE if name in PRECISE else 1.0)
        hist = []
        for k in range(n_max):
            p_now, _ = franka_fk(rt.q)
            hist.append(p_now)
            yaw_now = hand_yaw()
            dyaw = wrap(yaw_t - yaw_now)
            if name == "release":     # 指が開ききるまで待つ(開ききる前に上げると箱を引きずる)
                if k >= n_min and env.gripper_position() < 0.05:
                    break
            elif k >= n_min and np.linalg.norm(p_goal - p_now) < 0.005 and abs(dyaw) < np.deg2rad(3):
                break
            # 進まなくなったら次へ(箱が机に当たって降りきれない、など)
            if name != "release" and k >= max(n_min, 5) and np.linalg.norm(hist[-1] - hist[-6]) < 0.001:
                break
            if len(acts) >= max_steps:
                break
            d = p_goal - p_now
            p_t = p_now + d * min(1.0, lim / max(np.linalg.norm(d), 1e-9))
            yaw_cmd = yaw_now + np.clip(dyaw, -YAW_RATE, YAW_RATE)
            c, s_ = np.cos(yaw_cmd - psi0), np.sin(yaw_cmd - psi0)
            q_des = franka_ik(rt.q, p_t, np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ R0)
            a = np.concatenate([np.clip((q_des - rt.q) / DELTA, -1, 1), [g]])
            if render:
                ext.append(obs["observation/exterior_image_1_left"])
                wrist.append(obs["observation/wrist_image_left"])
            qs.append(rt.q.copy())
            grips.append(env.gripper_position())
            acts.append(a)
            phases.append(name)
            a_exec = a.copy()
            a_exec[:7] = np.clip(a[:7] + rng.normal(0, sigma, 7), -1, 1)
            env.step(a_exec)
            if render:
                obs = env.obs(task.prompt)
        phase_box[name] = np.round(env.box_pos(), 3).tolist()
    ok = place_success(env, task)
    box = env.box_pos()
    env.close()
    info = {"seed": int(seed), "success": ok, "steps": len(acts), "box_final": box.tolist(), "box_yaw": box_yaw,
            "hover": hover, "fast": fast, "dart_sigma": dart_sigma, "task": task.to_dict(),
            "phase_steps": {p: phases.count(p) for p in dict.fromkeys(phases)}, "phase_box": phase_box}
    data = None
    if render:
        data = dict(exterior_image_1_left=np.array(ext, np.uint8), wrist_image_left=np.array(wrist, np.uint8),
                    joint_position=np.array(qs, np.float32), gripper_position=np.array(grips, np.float32)[:, None],
                    actions=np.array(acts, np.float32), prompt=np.array(task.prompt),
                    phase=np.array(phases), task_json=np.array(json.dumps(task.to_dict())))
    return info, data


def worker(args):
    seed, out, check, sigma, force = args
    info, data = run_episode(seed, render=not check, dart_sigma=sigma, force=force)
    if not check and info["success"]:
        np.savez_compressed(os.path.join(out, f"ep_{seed:06d}.npz"), **data)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240, help="保存する成功エピソードの数")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--dart-sigma", type=float, default=DART_SIGMA)
    ap.add_argument("--out", default="data/demos")
    ap.add_argument("--check", action="store_true", help="描画も保存もせず、成功率だけを見る")
    ap.add_argument("--force", default=None, help='水準を指定(JSON)。例: \'{"box_color": "red", "target": "blue", "template": 1}\'')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    infos, seed, t0 = [], args.seed0, time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        while sum(i["success"] for i in infos) < args.n:
            need = args.n - sum(i["success"] for i in infos)
            batch = list(range(seed, seed + max(int(need * 1.2), args.workers)))
            seed += len(batch)
            force = json.loads(args.force) if args.force else None
            for info in pool.imap_unordered(worker, [(s, args.out, args.check, args.dart_sigma, force) for s in batch]):
                infos.append(info)
                t = info["task"]
                print(f"seed {info['seed']:6d} ok {info['success']:d} steps {info['steps']:3d} "
                      f"box_final {np.round(info['box_final'], 3)} target {np.round(t['target_xy'], 2)} "
                      f"phases {info['phase_steps']}  [{sum(i['success'] for i in infos)}/{len(infos)}, "
                      f"{time.time() - t0:.0f}s]", flush=True)
            if args.check:
                break
    infos.sort(key=lambda i: i["seed"])
    ok = [i for i in infos if i["success"]]
    print(f"success {len(ok)}/{len(infos)}, steps mean {np.mean([i['steps'] for i in ok]) if ok else 0:.0f}")
    if not args.check:
        name = "summary.json" if not args.force else f"summary_{args.seed0}.json"
        with open(os.path.join(args.out, name), "w") as f:
            json.dump({"dart_sigma": args.dart_sigma, "episodes": infos}, f, indent=1)


if __name__ == "__main__":
    main()
