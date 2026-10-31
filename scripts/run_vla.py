"""シミュで VLA(openpi の推論サーバ)を動かし、mp4 に保存する。
使い方: MUJOCO_GL=osmesa python scripts/run_vla.py --host HOST --port 8000 --prompt "pick up the orange cube" --steps 150 --out outputs/day3/zs.mp4
サーバが無いときの配線確認: --fake zero|random(行動を自前で作る)
DROID と同様に、サーバへ 1 回問い合わせて 10 ステップのチャンクを得て、先頭 8 ステップを実行する。
mp4 と同じ名前の .json に、引数・各ステップの行動・箱の高さ・最終状態を保存する。
--retarget: 行動を仮想の Franka に当て、その手先を OpenArm の IK で追う(条件 B。src/openarm_anomaly/franka_retarget.py)。
  既定では箱を (0.30, -0.20)、開始時の TCP を (0.18, -0.20, 0.28) に置く。"""
import argparse
import json
import sys
import time

import mediapy
import numpy as np

sys.path.insert(0, "src")
from openarm_anomaly.vla_env import CONTROL_HZ, DroidStyleEnv

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--prompt", default="pick up the orange cube")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--horizon", type=int, default=8)
ap.add_argument("--delta-scale", type=float, default=0.2)
ap.add_argument("--fake", choices=["zero", "random"])
ap.add_argument("--retarget", action="store_true")
ap.add_argument("--no-track-yaw", action="store_true", help="--retarget 時、Franka 側のヨーを追わない")
ap.add_argument("--box-xy", type=float, nargs=2, default=None)
ap.add_argument("--ready-tcp", type=float, nargs=3, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/day3/vla.mp4")
args = ap.parse_args()

if args.retarget:
    box_xy = tuple(args.box_xy or (0.30, -0.20))
    ready_tcp = tuple(args.ready_tcp or (0.18, -0.20, 0.28))
else:
    box_xy = tuple(args.box_xy or (0.18, -0.36))
    ready_tcp = tuple(args.ready_tcp) if args.ready_tcp else None
env = DroidStyleEnv(box_xy=box_xy, delta_scale=args.delta_scale, retarget=args.retarget,
                    ready_tcp=ready_tcp, track_yaw=not args.no_track_yaw)
rng = np.random.default_rng(args.seed)
if args.fake:
    policy = None
else:
    from openpi_client import websocket_client_policy
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)


def infer(obs):
    if policy is not None:
        return np.asarray(policy.infer(obs)["actions"])
    if args.fake == "zero":
        return np.zeros((10, 8))
    return rng.uniform(-1, 1, (10, 8)) * 0.5


obs = env.obs(args.prompt)
frames, pending = [], np.zeros((0, 8))
actions, box_z, tcp, n_infer = [], [], [], 0
box_z0 = float(env.box_pos()[2])
t0 = time.time()
for t in range(args.steps):
    if len(pending) == 0:
        chunk = infer(obs)
        n_infer += 1
        pending = chunk[:args.horizon]
    actions.append(np.asarray(pending[0], dtype=float).tolist())
    env.step(pending[0])
    box_z.append(float(env.box_pos()[2]))
    tcp.append(env.kin.fk(env.d.qpos[:7])[0].tolist())
    pending = pending[1:]
    obs = env.obs(args.prompt)
    frames.append(np.concatenate([obs["observation/exterior_image_1_left"], obs["observation/wrist_image_left"]], axis=1))
elapsed = time.time() - t0
print(f"{args.steps} steps in {elapsed:.1f}s, box={env.box_pos().round(3)}, grip={env.gripper_position():.2f}, "
      f"q={obs['observation/joint_position'].round(2)}")
mediapy.write_video(args.out, frames, fps=CONTROL_HZ)
env.close()
record = {
    "args": vars(args), "elapsed_s": elapsed, "n_infer": n_infer,
    "box_z0": box_z0, "box_z_max": max(box_z), "box_final": env.box_pos().tolist(),
    "gripper_final": env.gripper_position(), "q_final": obs["observation/joint_position"].tolist(),
    "q_openarm_final": env.d.qpos[:7].tolist(), "box_xy": box_xy, "ready_tcp": ready_tcp,
    "actions": actions, "box_z": box_z, "tcp": tcp,
}
json_path = args.out.rsplit(".", 1)[0] + ".json"
with open(json_path, "w") as f:
    json.dump(record, f)
print("saved", args.out, json_path)
