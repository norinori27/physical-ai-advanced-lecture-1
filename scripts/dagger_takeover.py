"""DAgger の正解(scripts/dagger.py の教師)の確認用動画: モデルがずれた所に降ろし始めた時点から、教師に交代させる。

記録(eval_vla.py の npz)から「箱を持ったまま、ゴールから 3〜10 cm ずれた所で、箱が机から 6 cm 以内まで降りた」最初の時点を探す。
その時点の腕・箱・仮想 Franka の状態を置き直し(腕と箱を留めたまま指を閉じて握り直させる)、そこから教師で動かす。
左: モデルがそのまま続けた記録、右: 教師に交代した場合。上は外部カメラ、下は手首カメラ。交代の 2 秒前から映す。
使い方: MUJOCO_GL=osmesa python scripts/dagger_takeover.py --src outputs/eval_npz/dagger1_p0/dagger1_p0 [--max 6]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import mediapy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from dagger import Context, env_state, teacher  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PRE = 30          # 交代の前に映すフレーム数(2 秒)


def find_takeover(z, h):
    task = json.loads(str(z["task_json"]))
    tg = np.asarray(task["target_xy"])
    box, pad, grip = z["box_pos"], z["pad_contact"], np.ravel(z["gripper_position"])
    for t in range(1, len(box)):
        e = np.linalg.norm(box[t - 1, :2] - tg)
        if pad[t - 1] > 0 and grip[t] > 0.1 and 0.03 < e < 0.10 and box[t - 1, 2] < h + 0.06:
            return t, float(e)
    return None, None


def run(path, out_dir, steps):
    from openarm_anomaly.task import apply_task, place_success, sample_task
    from openarm_anomaly.vla_env import DroidStyleEnv
    z = np.load(path)
    task_d = json.loads(str(z["task_json"]))
    seed = int(os.path.basename(path).rsplit("_s", 1)[1][:-4])
    task = sample_task(np.random.default_rng(seed), task_d["condition"], seed)
    env = DroidStyleEnv(box_xy=task.box_xy, retarget=True, ready_tcp=task.ready_tcp, track_yaw=True, markers=True)
    apply_task(env, task)
    m, d = env.m, env.d
    t0, err0 = find_takeover(z, env.box_half)
    if t0 is None:
        env.close()
        return None
    ctx = Context(env.kin, task.to_dict(), z["q_openarm"][0], z["joint_position"][0], env.rt.offset, env.box_half)
    # 状態を再現(観測 t0 の時点 = 行動 t0-1 の後)。記録した行動を実行し直す方法は、Modal と手元の計算のわずかな差が
    # 接触で広がり、箱が 10 cm 以上ずれたので使わない。直接置き直し、腕と箱をその場に留めたまま指を閉じて握り直させる
    import mujoco
    from openarm_anomaly.scripted import GRIP_CLOSE, GRIP_OPEN, N_ARM
    jb = m.joint("box_free").qposadr[0]
    q = z["q_openarm"][t0 - 1]
    finger = GRIP_OPEN - float(np.ravel(z["gripper_position"])[t0]) * (GRIP_OPEN - GRIP_CLOSE)
    d.qpos[N_ARM:N_ARM + 2] = finger + 0.004       # 少し開いた所から閉じる(めり込みで弾かないように)
    for _ in range(80):
        d.qpos[:N_ARM] = q
        d.qpos[jb:jb + 3] = z["box_pos"][t0 - 1]
        d.qpos[jb + 3:jb + 7] = z["box_quat"][t0 - 1]
        d.qvel[:N_ARM] = 0
        d.qvel[m.joint("box_free").dofadr[0]:][:6] = 0
        d.ctrl[:N_ARM] = q
        d.ctrl[N_ARM] = GRIP_CLOSE
        mujoco.mj_step(m, d)
    d.qpos[:N_ARM] = q
    d.qpos[jb:jb + 3] = z["box_pos"][t0 - 1]
    d.qpos[jb + 3:jb + 7] = z["box_quat"][t0 - 1]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    env.q_cmd = q.copy()
    # 評価のときの IK の出発点(前のステップの指令)は記録していないので、実際の関節角から始める。ふだんと違う姿勢では
    # 出発点の違いで別の解に飛んで腕が跳ねたので、1 ステップの関節の変化に上限を付ける(この確認動画のシミュだけ)
    env.max_dq = 0.15
    env.rt.q = z["joint_position"][t0].astype(float).copy()
    env.rt.q_cmd = env.rt.q.copy()
    pads = {m.geom("openarm_left_finger_pad").id, m.geom("openarm_right_finger_pad").id}
    box_gid = m.geom("box_geom").id
    frames, names, success_t = [], [], None
    for k in range(steps):
        obs = env.obs("")
        frames.append(np.concatenate([obs["observation/exterior_image_1_left"], obs["observation/wrist_image_left"]], 0))
        a, name = teacher(ctx, env.kin, env_state(env, pads, box_gid))
        names.append(name)
        env.step(a)
        s = env_state(env, pads, box_gid)
        if success_t is None and place_success(env, task) and s["pad"] == 0 and s["tcp"][2] - s["box"][2] > 0.05:
            success_t = k
        if success_t is not None and k > success_t + 15:
            break
    box_t = env.box_pos()
    env.close()
    dist_teacher = float(np.hypot(*(box_t[:2] - np.asarray(task.target_xy))))
    dist_model = float(np.hypot(*(z["box_pos"][-1][:2] - np.asarray(task.target_xy))))

    # 動画: 左 = モデル(記録)、右 = 教師に交代。各マスは 320x180 を 1.5 倍
    model = np.concatenate([z["ext"], z["wrist"]], 1)        # (T, 360, 320, 3)
    T = PRE + max(len(frames), min(len(model) - t0, len(frames) + 30))
    W, H, cap = 2 * 480 + 10, 540 + 70, 70
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(path).replace(".npz", "_takeover.mp4"))
    base = Image.new("RGB", (W, H), (252, 252, 251))
    dr = ImageDraw.Draw(base)
    dr.text((8, 4), f'{os.path.basename(path)[:-4]}: "{task_d["prompt"]}"   takeover at t={t0} (cube {err0 * 100:.1f} cm off the goal)',
            font=ImageFont.truetype(FONT_B, 15), fill=(11, 11, 11))
    dr.text((8, 28), f"left: model continues -> {dist_model * 100:.1f} cm from goal at end", font=ImageFont.truetype(FONT, 14), fill=(82, 81, 78))
    dr.text((W // 2 + 5, 28), f"right: DAgger teacher takes over -> {dist_teacher * 100:.1f} cm"
            f" ({'placed, released, lifted' if success_t is not None else 'not finished'})", font=ImageFont.truetype(FONT, 14), fill=(82, 81, 78))
    dr.text((8, 48), "(top: exterior camera, bottom: wrist camera; 2 s before takeover are identical)", font=ImageFont.truetype(FONT, 12), fill=(82, 81, 78))
    canvas0 = np.asarray(base).copy()
    with mediapy.VideoWriter(out, shape=(H, W), fps=15) as w:
        for i in range(T):
            f = canvas0.copy()
            tm = min(t0 - PRE + i, len(model) - 1)
            left = model[max(tm, 0)]
            right = model[t0 - PRE + i] if i < PRE else frames[min(i - PRE, len(frames) - 1)]
            for j, img in enumerate((left, right)):
                f[cap:cap + 540, j * 490:j * 490 + 480] = cv2.resize(img, (480, 540))
            w.add_image(f)
    return {"file": os.path.basename(path), "t0": t0, "err0_cm": round(err0 * 100, 1), "model_end_cm": round(dist_model * 100, 1),
            "teacher_end_cm": round(dist_teacher * 100, 1), "teacher_success": success_t is not None,
            "teacher_phases": {p: names.count(p) for p in dict.fromkeys(names)}, "video": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="outputs/video/dagger_check")
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    n = 0
    for f in sorted(glob.glob(os.path.join(args.src, "*.npz"))):
        r = run(f, args.out, args.steps)
        if r is None:
            continue
        print(r, flush=True)
        n += 1
        if n >= args.max:
            break
    print(f"{n} videos in {args.out}")


if __name__ == "__main__":
    main()
