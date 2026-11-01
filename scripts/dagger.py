"""DAgger(Ross et al., AISTATS 2011): モデルがたどり着いた状態に、教師の行動を正解として付ける。

gen_demos.py の教師は段階を順に進むので、途中の状態からは使えない。ここでは「今の状態から次の行き先を決める」教師を作る。
行動の計算(経由点 → 仮想 Franka の IK → (目標 − 現在) / 0.2、ヨーは 1 ステップ 4° まで)は gen_demos.py と同じ。

  箱を持っている
    ゴールから水平に 1 cm 以上ずれている → 降ろさずに横へ動いて直す(5 cm 以上なら運搬の高さで、近ければ今の高さで)
    ずれが 1 cm 以内                    → 降ろす。机に着いたら放す
  箱を持っていない
    箱がゴールのマーカーの中で机の上   → 指が開くまで待ってから、手を上げる
    それ以外(つかむ前、落とした、外に置いた) → 箱の真上へ行き、向きを合わせて降り、つかむ(置き直しを含む)

使い方:
  確認(教師だけで動かす): MUJOCO_GL=osmesa python scripts/dagger.py check --n 24 [--perturb]
    --perturb: 運搬中に実行だけをゴールから 3〜8 cm ずらした点へ向け(正解のラベルは付けない)、直せるかを見る
  正解を付ける: python scripts/dagger.py relabel --src outputs/eval_npz/dagger1 --out data/demos_dagger1
    評価の npz(eval_vla.py)から、gen_demos.py と同じ形の ep_<seed>.npz を作る(画像と状態はモデルが見たもの、行動は教師)
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

from gen_demos import DELTA, LIFT  # noqa: E402

HOVER, FAST, SLOW = 0.13, 0.016, 0.006
YAW_RATE = np.deg2rad(4)
HOLD_YAW = os.environ.get("DAGGER_HOLD_YAW", "goal")
YAW_TOL = np.deg2rad(12)   # 箱の向きとのずれがこれ以内なら降りる(手首が回りきらず届かない向きがある。指は 10° 程度のずれを吸収する)
LATERAL_TOL = 0.01         # 箱を持っているとき、これ以上ずれていたら降ろさずに直す(m)
FAR = 0.05                 # これ以上ずれていたら運搬の高さに戻して運ぶ
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def quat_yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Context:
    """エピソードで変わらない量(gen_demos.py の run_episode の最初と同じ計算)。"""

    def __init__(self, kin, task, q_oa0, q_f0, offset, box_half):
        from openarm_anomaly.franka_retarget import franka_fk
        self.task, self.h, self.offset = task, box_half, np.asarray(offset)
        self.R0 = franka_fk(q_f0)[1]
        self.psi0 = yaw_of(kin.fk(q_oa0)[1])
        q_g = kin.solve(np.asarray(q_oa0, float).copy(), np.array([*task["box_xy"], box_half + LIFT]), iters=100)
        q_g = kin.solve(q_g, np.array([*task["target_xy"], box_half + LIFT]), iters=100)
        q_g = kin.solve(q_g, np.array([*task["target_xy"], box_half]), iters=100)
        self.yaw_goal_nat = yaw_of(kin.fk(q_g)[1])
        self.goal = np.array([*task["target_xy"], box_half + 0.005])   # 置くときの箱の中心(世界座標)


def teacher(ctx, kin, s):
    """s: q_f(仮想 Franka の関節角)、q_oa、grip(0 開 〜 1 閉)、box(箱の中心、世界)、box_quat、tcp(世界)、pad(指と箱の接触数)。
    返り値: (行動 8, 段階の名前)"""
    from openarm_anomaly.franka_retarget import franka_fk, franka_ik
    from openarm_anomaly.task import MARKER_HALF
    p_now = franka_fk(s["q_f"])[0]
    yaw_now = yaw_of(kin.fk(s["q_oa"])[1])
    box, tcp, h = np.asarray(s["box"]), np.asarray(s["tcp"]), ctx.h
    box_yaw = quat_yaw(s["box_quat"])
    # gen_demos.py と同じく、箱の向きそのもの(開始時のハンドの向きに近い側の巻き)を狙う。90° ずらした等価な向きは手首が回りきらないことがある
    yaw_grasp = ctx.psi0 + wrap(box_yaw - ctx.psi0)
    yaw_goal = yaw_grasp + wrap(ctx.yaw_goal_nat - yaw_grasp)
    holding = s["pad"] > 0 and s["grip"] > 0.1
    resting = box[2] < h + 0.008
    e = ctx.goal[:2] - box[:2]
    on_target = abs(e[0]) < MARKER_HALF and abs(e[1]) < MARKER_HALF

    move = None        # 手先の移動(世界座標の差)。gen_demos と同じく、手先と箱の中心を同じ点とみなさず、差で動かす
    if holding:
        yaw_t = yaw_goal if HOLD_YAW == "goal" else yaw_now
        if np.linalg.norm(e) > LATERAL_TOL:
            z_t = h + LIFT if np.linalg.norm(e) > FAR else max(box[2], h + 0.03)
            move, g, lim, name = np.array([e[0], e[1], z_t - box[2]]), 1.0, FAST if np.linalg.norm(e) > FAR else SLOW, "correct"
        elif not resting:
            move, g, lim, name = ctx.goal - box, 1.0, SLOW, "lower"
        else:
            move, g, lim, name = np.zeros(3), 0.0, SLOW, "release"
    elif on_target and resting:
        yaw_t = yaw_now
        if s["grip"] > 0.05:                     # 指が開ききるまで待つ(開ききる前に上げると箱を引きずる)
            move, g, lim, name = np.zeros(3), 0.0, SLOW, "release"
        elif tcp[2] - box[2] < 0.04:
            move, g, lim, name = np.array([0, 0, box[2] + 0.045 - tcp[2]]), 0.0, SLOW, "retreat_slow"
        else:
            move, g, lim, name = np.array([0, 0, box[2] + 0.10 - tcp[2]]), 0.0, FAST, "retreat"
    else:                                         # つかむ(最初の把持、落とした箱、外に置いた箱)
        yaw_t = yaw_grasp
        dxy = box[:2] - tcp[:2]
        dz = tcp[2] - box[2]
        aligned = np.linalg.norm(dxy) < 0.008 and abs(wrap(yaw_grasp - yaw_now)) < YAW_TOL
        if s["grip"] > 0.3 and not holding:       # つかみ損ねて閉じている: 開きながら上へ
            move, g, lim, name = np.array([0, 0, max(0.0, box[2] + 0.05 - tcp[2])]), 0.0, SLOW, "reopen"
        elif not aligned:
            if dz < 0.035 and np.linalg.norm(dxy) > 0.015:   # 箱の近くで低い: 箱を押さないよう、まず上へ
                move, g, lim, name = np.array([0, 0, box[2] + 0.04 - tcp[2]]), 0.0, SLOW, "up"
            else:
                z_t = box[2] + (HOVER if np.linalg.norm(dxy) > 0.03 else 0.04)
                move, g, lim, name = np.array([dxy[0], dxy[1], z_t - tcp[2]]), 0.0, FAST, "approach"
        elif dz > 0.004:
            move, g, lim, name = np.array([dxy[0], dxy[1], -dz]), 0.0, SLOW, "descend"
        else:
            move, g, lim, name = np.zeros(3), 1.0, SLOW, "grasp"

    p_goal = p_now + move
    d = p_goal - p_now
    p_t = p_now + d * min(1.0, lim / max(np.linalg.norm(d), 1e-9))
    yaw_cmd = yaw_now + np.clip(wrap(yaw_t - yaw_now), -YAW_RATE, YAW_RATE)
    c, s_ = np.cos(yaw_cmd - ctx.psi0), np.sin(yaw_cmd - ctx.psi0)
    q_des = franka_ik(s["q_f"], p_t, np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ ctx.R0)
    a = np.concatenate([np.clip((q_des - s["q_f"]) / DELTA, -1, 1), [g]])
    return a, name


def env_state(env, pads, box_gid):
    d = env.d
    jb = env.m.joint("box_free").qposadr[0]
    return {"q_f": env.rt.q.copy(), "q_oa": d.qpos[:7].copy(), "grip": env.gripper_position(),
            "box": d.qpos[jb:jb + 3].copy(), "box_quat": d.qpos[jb + 3:jb + 7].copy(), "tcp": env.kin.fk(d.qpos[:7])[0],
            "pad": sum(1 for c in d.contact[:d.ncon] if {c.geom1, c.geom2} & pads and box_gid in (c.geom1, c.geom2))}


def check_one(job):
    """教師だけで動かす。perturb: 運搬中の 25 ステップだけ、実行する行動をゴールから 3〜8 cm ずれた点へ向ける。"""
    seed, perturb, max_steps = job
    from openarm_anomaly.task import apply_task, place_success, sample_task
    from openarm_anomaly.vla_env import DroidStyleEnv
    rng = np.random.default_rng(seed)
    task = sample_task(rng, "train", seed)
    env = DroidStyleEnv(box_xy=task.box_xy, retarget=True, ready_tcp=task.ready_tcp, track_yaw=True, markers=True)
    apply_task(env, task)
    m = env.m
    pads = {m.geom("openarm_left_finger_pad").id, m.geom("openarm_right_finger_pad").id}
    box_gid = m.geom("box_geom").id
    ctx = Context(env.kin, task.to_dict(), env.d.qpos[:7].copy(), env.rt.q.copy(), env.rt.offset, env.box_half)
    if perturb:
        ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(0.03, 0.08)
        fake = Context(env.kin, {**task.to_dict(), "target_xy": list(np.asarray(task.target_xy) + r * np.array([np.cos(ang), np.sin(ang)]))},
                       env.d.qpos[:7].copy(), env.rt.q.copy(), env.rt.offset, env.box_half)
    names, n_pert, ok_run, success = [], 0, 0, False
    for t in range(max_steps):
        s = env_state(env, pads, box_gid)
        a, name = teacher(ctx, env.kin, s)
        if perturb and name in ("correct", "lower") and n_pert < 25:
            a, _ = teacher(fake, env.kin, s)       # 実行だけずらす(DAgger では、この状態にも正しい行動が付く)
            n_pert += 1
        names.append(name)
        env.step(a)
        placed = place_success(env, task)
        s2 = env_state(env, pads, box_gid)
        released = placed and s2["pad"] == 0 and s2["tcp"][2] - s2["box"][2] >= 0.05
        ok_run = ok_run + 1 if placed else 0
        if ok_run >= 8 and released:
            success = True
            break
    box = env.box_pos()
    env.close()
    phases = {p: names.count(p) for p in dict.fromkeys(names)}
    return {"seed": seed, "success": success, "steps": t + 1, "phases": phases,
            "dist": float(np.hypot(*(box[:2] - np.asarray(task.target_xy))))}


def relabel_one(job):
    path, out = job
    from openarm_anomaly.franka_retarget import franka_fk
    from openarm_anomaly.vla_env import DroidStyleEnv
    z = np.load(path)
    task = json.loads(str(z["task_json"]))
    env = DroidStyleEnv(box_xy=task["box_xy"], retarget=True, ready_tcp=task["ready_tcp"], track_yaw=True, markers=True)
    kin, h = env.kin, env.box_half
    q_f, q_oa, grip = z["joint_position"], z["q_openarm"], np.ravel(z["gripper_position"])
    n = len(z["action"])
    # 仮想 Franka の座標と世界座標の差。記録の tcp[t] は行動 t の後、joint_position[t+1] はその後の観測
    off = np.median([z["tcp"][t] - franka_fk(q_f[t + 1])[0] for t in range(0, n - 1, max(1, (n - 1) // 30))], 0)
    # 観測 t の時点の状態: 箱・手先・接触・OpenArm の関節角は行動 t-1 の後の記録(t = 0 は行動 0 の後で近似。まだ箱は動いていない)
    ctx = Context(kin, task, q_oa[0], q_f[0], off, h)
    acts, names = [], []
    for t in range(n):
        p = max(t - 1, 0)
        s = {"q_f": q_f[t], "q_oa": q_oa[p], "grip": float(grip[t]), "box": z["box_pos"][p], "box_quat": z["box_quat"][p],
             "tcp": z["tcp"][p], "pad": int(z["pad_contact"][p])}
        a, name = teacher(ctx, kin, s)
        acts.append(a)
        names.append(name)
    env.close()
    seed = int(os.path.basename(path).rsplit("_s", 1)[1][:-4])
    np.savez_compressed(os.path.join(out, f"ep_{seed:06d}.npz"),
                        exterior_image_1_left=z["ext"], wrist_image_left=z["wrist"],
                        joint_position=q_f.astype(np.float32), gripper_position=grip.astype(np.float32)[:, None],
                        actions=np.array(acts, np.float32), prompt=np.array(task["prompt"]), phase=np.array(names),
                        task_json=np.array(json.dumps(task)), model_action=z["action"].astype(np.float32))
    a_m, a_t = z["action"][:, :7], np.array(acts)[:, :7]
    return {"file": os.path.basename(path), "n": n, "phases": {p: names.count(p) for p in dict.fromkeys(names)},
            "mean_abs_diff_model_vs_teacher": float(np.mean(np.abs(np.clip(a_m, -1, 1) - a_t)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "relabel"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=800000)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--src", nargs="+", default=[])
    ap.add_argument("--out", default="data/demos_dagger1")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    with get_context("spawn").Pool(args.workers) as pool:
        if args.mode == "check":
            res = list(pool.imap_unordered(check_one, [(args.seed0 + i, args.perturb, args.max_steps) for i in range(args.n)]))
            for r in sorted(res, key=lambda r: r["seed"]):
                print(r, flush=True)
            print(f"success {sum(r['success'] for r in res)}/{len(res)} (perturb={args.perturb})")
        else:
            os.makedirs(args.out, exist_ok=True)
            files = sorted(f for d in args.src for f in glob.glob(os.path.join(d, "*.npz")))
            for r in pool.imap_unordered(relabel_one, [(f, args.out) for f in files]):
                print(r, flush=True)
            print(f"saved {len(files)} episodes to {args.out}")


if __name__ == "__main__":
    main()
