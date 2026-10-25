"""Day 1: OpenArm 単腕モデルを MuJoCo / MJX で読み込めるか確認する。"""
import sys
import time

import mujoco
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/openarm_mujoco/v1/openarm.xml"
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
print(f"nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom} nmesh={m.nmesh} neq={m.neq} ntendon={m.ntendon}")
print("timestep", m.opt.timestep, "integrator", m.opt.integrator)

# 衝突に使うメッシュの頂点数(MJX の凸包の制約を確認)
for g in range(m.ngeom):
    if m.geom_contype[g] or m.geom_conaffinity[g]:
        mid = m.geom_dataid[g]
        nv = m.mesh_vertnum[mid] if mid >= 0 else 0
        print(f"  collision geom {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g):35s} mesh verts={nv}")

mujoco.mj_resetDataKeyframe(m, d, 0)
t = time.time()
for _ in range(1000):
    mujoco.mj_step(m, d)
print(f"CPU MuJoCo 1000 steps: {time.time()-t:.3f}s, qpos={np.round(d.qpos, 3)}")

from mujoco import mjx
try:
    mx = mjx.put_model(m)
    print("mjx.put_model: OK")
except Exception as e:
    print("mjx.put_model FAILED:", type(e).__name__, e)
