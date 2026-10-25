"""Kaggle 用の Day 1 確認ノートブックを生成する。

src/openarm_anomaly/ の中身を %%writefile セルとして埋め込むので、
手元のコードを変えたら再実行してノートブックを作り直す。
    python scripts/build_kaggle_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_day1_check.ipynb"
OPENARM_COMMIT = "a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c"
SRC_FILES = ["__init__.py", "scene.py", "scripted.py"]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip()})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip()})


md("""
# OpenArm 違和感検知: Day 1 確認(Kaggle GPU)

PLAN.md 第 6 節の 4・5 を確認する。

- 4: Kaggle 環境(ヘッドレス描画、JAX/MJX の GPU 動作、mp4 書き出し)
- 5: MJX の並列速度から PPO の学習時間を見積もる

**設定**: Accelerator = GPU T4(x1 で可)、Internet = On。上から順に実行する。
結果は最後のセルで `/kaggle/working/day1_results.json` に保存される。
""")

code("!nvidia-smi")

code("""
%%capture pip_log
!pip install -q "jax[cuda12]==0.10.2" mujoco==3.17.0 mujoco-mjx==3.17.0 warp-lang==1.18.0 \\
    brax==0.14.2 playground==0.2.0 flax==0.12.8 optax==0.2.8 mediapy==1.2.7 "imageio[ffmpeg]"
""")

code("""
print(pip_log.stdout[-3000:], pip_log.stderr[-3000:])
""")

code(f"""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.chdir("/kaggle/working")
!rm -rf openarm_mujoco && git clone -q https://github.com/enactic/openarm_mujoco.git && \\
    git -C openarm_mujoco checkout -q {OPENARM_COMMIT} && git -C openarm_mujoco log -1 --format=%H
!mkdir -p src/openarm_anomaly outputs
""")

for name in SRC_FILES:
    body = (ROOT / "src" / "openarm_anomaly" / name).read_text()
    code(f"%%writefile src/openarm_anomaly/{name}\n{body}")

code("""
import json, os, sys, time
sys.path.insert(0, "src")
import jax, jax.numpy as jnp, mujoco, numpy as np
from mujoco import mjx
from openarm_anomaly.scene import STATION_B, build_model
from openarm_anomaly.scripted import ScriptedController, reset_to_ready, run_episode

print("jax", jax.__version__, jax.devices(), "mujoco", mujoco.__version__)
assert jax.devices()[0].platform == "gpu", "GPU が見えていない"
results = {"jax": jax.__version__, "device": str(jax.devices()[0]), "mujoco": mujoco.__version__}
""")

md("""
## A. MJX(GPU)で保持が成立するか

CPU MuJoCo のスクリプト制御で指令列を記録し、同じ指令列を MJX で再生して箱の軌跡を比べる。
物理の時間刻み 0.002 s(基準)と 0.004 s(学習を速くできるか)の両方を、
`impl="jax"` と `impl="warp"`(MJWarp)で試す。Warp は接触数の上限を事前に確保するため、
上限が足りないと接触が落ちる。保持が CPU と一致するかで確かめる。
""")

code("""
HALF, MASS, CTRL_DT = 0.03, 1.5, 0.02
NACON_PER_ENV, NJMAX = 32, 256  # warp 用の接触数・拘束行数の上限


def batched_data(m, d0, impl, B):
    if impl == "jax":
        dx = mjx.put_data(m, d0)
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), dx)
    return jax.vmap(lambda i: mjx.put_data(
        m, d0, impl="warp", dummy_arg_for_batching=i,
        naconmax=B * NACON_PER_ENV, njmax=NJMAX))(jnp.arange(B))


def record(dt):
    m = build_model(box_half=HALF, box_mass=MASS, timestep=dt)
    d = mujoco.MjData(m)
    reset_to_ready(m, d, HALF)
    d0 = mujoco.MjData(m)
    d0.qpos[:], d0.ctrl[:] = d.qpos, d.ctrl
    mujoco.mj_forward(m, d0)
    log, _ = run_episode(m, d, ScriptedController(m, HALF, ctrl_dt=CTRL_DT))
    return m, d0, log


def make_rollout(m, mx, impl_tag=""):
    n_sub = int(round(CTRL_DT / m.opt.timestep))
    box_id = m.body("box").id

    def ctrl_step(dx, u):
        dx = dx.replace(ctrl=u)
        dx = jax.lax.fori_loop(0, n_sub, lambda _, x: mjx.step(mx, x), dx)
        return dx, dx.xpos[box_id]
    return ctrl_step, n_sub


results["replay"] = {}
for impl in ("jax", "warp"):
    for dt in (0.002, 0.004):
        key = f"{impl}_dt{dt}"
        try:
            m, d0, log = record(dt)
            mx = mjx.put_model(m, impl=impl)
            dxb = batched_data(m, d0, impl, 1)
            ctrl_step, n_sub = make_rollout(m, mx)
            roll = jax.jit(jax.vmap(lambda dx, us: jax.lax.scan(ctrl_step, dx, us)[1], in_axes=(0, None)))
            ctrls = jnp.asarray(log["ctrl"])
            t = time.time(); box = np.asarray(jax.block_until_ready(roll(dxb, ctrls)))[0]; t_compile = time.time() - t
            cpu_err = float(np.linalg.norm(log["box_pos"][-1, :2] - np.array(STATION_B)))
            mjx_err = float(np.linalg.norm(box[-1, :2] - np.array(STATION_B)))
            r = {"cpu_place_err": cpu_err, "cpu_lift": float(log["box_pos"][:, 2].max() - HALF),
                 "mjx_place_err": mjx_err, "mjx_lift": float(box[:, 2].max() - HALF),
                 "compile_s": t_compile}
            print(f"{key}: {json.dumps({k: round(v, 4) for k, v in r.items()})}")
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"{key}: FAILED {r['error']}")
        results["replay"][key] = r
""")

md("""
## B. 並列速度(PPO の学習時間の見積もり)

同じ指令列を B 個の環境で並列に再生し、制御ステップ/秒を測る(方策の計算は含まない)。
PPO では 1 制御ステップ = 方策の 1 行動。`impl="jax"` と `impl="warp"`(MJWarp)を比べる。
""")

code("""
N_CTRL = 50  # 計測する制御ステップ数(1 秒分)
TARGET_STEPS = 100e6  # 見積もりの基準: PPO 1 億制御ステップ
results["throughput"] = []


def bench(dt, B, impl):
    m, d0, log = record(dt)
    mx = mjx.put_model(m, impl=impl)
    ctrl_step, n_sub = make_rollout(m, mx)
    ctrls = jnp.asarray(log["ctrl"][:N_CTRL])
    dxb = batched_data(m, d0, impl, B)
    f = jax.jit(jax.vmap(lambda x: jax.lax.scan(ctrl_step, x, ctrls)[1][-1]))
    t = time.time(); jax.block_until_ready(f(dxb)); t_compile = time.time() - t
    t = time.time(); last = np.asarray(jax.block_until_ready(f(dxb))); el = time.time() - t
    sps = B * N_CTRL / el
    r = {"impl": impl, "dt": dt, "B": B, "ctrl_steps_per_s": sps,
         "physics_steps_per_s": sps * n_sub, "compile_s": t_compile,
         "hours_per_100M": TARGET_STEPS / sps / 3600,
         "finite": bool(np.all(np.isfinite(last)))}
    results["throughput"].append(r)
    print(f"{impl:5s} dt={dt} B={B:5d}: {sps:10,.0f} ctrl steps/s  "
          f"-> {r['hours_per_100M']:.2f} h / 1e8 steps  (compile {t_compile:.0f}s, finite={r['finite']})")


for impl in ("jax", "warp"):
    for dt in (0.002, 0.004):
        for B in (256, 1024, 4096):
            try:
                bench(dt, B, impl)
            except Exception as e:  # warp が使えない、メモリ不足など
                print(f"{impl} dt={dt} B={B}: FAILED {type(e).__name__}: {str(e)[:300]}")
                results["throughput"].append({"impl": impl, "dt": dt, "B": B, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                break
""")

md("""
## C. ヘッドレス描画(EGL)と mp4 書き出し
""")

code("""
import mediapy
m = build_model(box_half=HALF, box_mass=1.0)
d = mujoco.MjData(m)
reset_to_ready(m, d, HALF)
cam = mujoco.MjvCamera()
cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [0.2, 0, 0.12], 1.0, 160, -22
r = mujoco.Renderer(m, 480, 640)
t = time.time()
log, frames = run_episode(m, d, ScriptedController(m, HALF), render=r, every=2, camera=cam)
t_render = time.time() - t
mediapy.write_video("outputs/pick_place_kaggle.mp4", frames, fps=25)
results["render"] = {"frames": len(frames), "seconds": t_render, "mp4_bytes": os.path.getsize("outputs/pick_place_kaggle.mp4")}
print(results["render"])
mediapy.show_video(frames, fps=25)
""")

code("""
with open("/kaggle/working/day1_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"wrote {OUT.relative_to(ROOT)} ({len(cells)} cells)")
