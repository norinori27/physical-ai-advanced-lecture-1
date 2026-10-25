"""Day 1: スクリプト制御で質量・摩擦・サイズの範囲内/外のピック&運搬が成立するか確認する。"""
import itertools
import sys

import mujoco
import numpy as np

sys.path.insert(0, "src")
from openarm_anomaly.scene import STATION_B, build_model
from openarm_anomaly.scripted import ScriptedController, reset_to_ready, run_episode

dt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
print(f"timestep={dt}")
for mass, fr, half in itertools.product([0.1, 1.5, 3.0], [0.5, 1.2], [0.025, 0.04]):
    m = build_model(box_half=half, box_mass=mass, friction=fr, timestep=dt)
    d = mujoco.MjData(m)
    reset_to_ready(m, d, half)
    ctrl = ScriptedController(m, half)
    log, _ = run_episode(m, d, ctrl)
    bp = log["box_pos"]
    lifted = bp[:, 2].max() - half
    err = np.linalg.norm(bp[-1, :2] - np.array(STATION_B))
    tau = np.abs(log["torque"]).max(0)
    ok = err < 0.02 and abs(bp[-1, 2] - half) < 0.005
    track = np.linalg.norm(log["tcp"] - log["tcp_des"], axis=1).max()
    print(f"track_max={track:.3f} m={mass:3.1f} mu={fr:.1f} half={half:.3f} lift={lifted:.3f} place_err={err:.3f} "
          f"ok={ok} |tau|max={np.round(tau, 1)}")
