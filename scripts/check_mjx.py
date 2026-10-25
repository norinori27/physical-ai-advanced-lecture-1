"""Day 1: CPU で記録した指令列を MJX で再生し、保持の成立と速度を確認する。"""
import sys
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

sys.path.insert(0, "src")
from openarm_anomaly.scene import STATION_B, build_model
from openarm_anomaly.scripted import ScriptedController, reset_to_ready, run_episode

print("jax", jax.__version__, jax.devices())
dt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
half, mass = 0.03, 1.5
m = build_model(box_half=half, box_mass=mass, timestep=dt)
d = mujoco.MjData(m)
reset_to_ready(m, d, half)
d0 = mujoco.MjData(m)
d0.qpos[:], d0.ctrl[:] = d.qpos, d.ctrl
mujoco.mj_forward(m, d0)
log, _ = run_episode(m, d, ScriptedController(m, half))
ctrls = jnp.asarray(log["ctrl"])
print(f"CPU: box final {log['box_pos'][-1].round(3)}  max lift {log['box_pos'][:, 2].max() - half:.3f}")

mx = mjx.put_model(m)
dx = mjx.put_data(m, d0)
n_sub = int(round(0.02 / dt))


def ctrl_step(dx, u):
    dx = dx.replace(ctrl=u)
    dx = jax.lax.fori_loop(0, n_sub, lambda _, x: mjx.step(mx, x), dx)
    return dx, (dx.xpos[m.body("box").id], dx.actuator_force[:7])


rollout = jax.jit(lambda dx, us: jax.lax.scan(ctrl_step, dx, us))
t = time.time()
_, (box, tau) = jax.block_until_ready(rollout(dx, ctrls))
print(f"MJX compile+run: {time.time() - t:.1f}s")
t = time.time()
_, (box, tau) = jax.block_until_ready(rollout(dx, ctrls))
el = time.time() - t
print(f"MJX run (1 env, {len(ctrls)} ctrl steps = {len(ctrls) * n_sub} physics steps): {el:.2f}s")
box = np.asarray(box)
err = np.linalg.norm(box[-1, :2] - np.array(STATION_B))
print(f"MJX: box final {box[-1].round(3)}  max lift {box[:, 2].max() - half:.3f}  place_err {err:.3f}")
print(f"torque diff CPU vs MJX (max abs): {np.abs(np.asarray(tau) - log['torque']).max():.2f}")

# バッチ(vmap)の速度
for B in (16, 64):
    dxb = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), dx)
    f = jax.jit(jax.vmap(lambda x: jax.lax.scan(ctrl_step, x, ctrls[:50])[0]))
    jax.block_until_ready(f(dxb))
    t = time.time()
    jax.block_until_ready(f(dxb))
    el = time.time() - t
    print(f"vmap B={B}: {B * 50 * n_sub / el:,.0f} physics steps/s on CPU")
