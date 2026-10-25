"""Day 1: スクリプト制御のピック&運搬 1 サイクルを mp4 にする。"""
import sys

import mediapy
import mujoco
import numpy as np

sys.path.insert(0, "src")
from openarm_anomaly.scene import STATION_B, build_model
from openarm_anomaly.scripted import ScriptedController, reset_to_ready, run_episode

half, mass = 0.03, 1.0
m = build_model(box_half=half, box_mass=mass)
d = mujoco.MjData(m)
reset_to_ready(m, d, half)
m.light_diffuse[:] = 0.9
cam = mujoco.MjvCamera()
cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [0.2, 0, 0.12], 1.0, 160, -22
r = mujoco.Renderer(m, 480, 640)
ctrl = ScriptedController(m, half)
log, frames = run_episode(m, d, ctrl, render=r, every=2, camera=cam)
fps = 1 / (ctrl.ctrl_dt * 2)
mediapy.write_video("outputs/day1/pick_place.mp4", frames, fps=fps)
err = np.linalg.norm(log["box_pos"][-1, :2] - np.array(STATION_B))
print(f"{len(frames)} frames @ {fps:.0f} fps, place_err={err:.3f}")
