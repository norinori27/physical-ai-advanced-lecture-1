"""gen_demos.py の npz を、openpi の DROID 形式の LeRobot データセットにする(openpi の環境で実行)。

項目は openpi の examples/droid/convert_droid_data_to_lerobot.py と同じ(画像は 180x320、15 fps、行動は関節速度 7 + グリッパ 1)。
exterior_image_2_left は openpi の LeRobotDROIDDataConfig の repack で必須だが、モデルは使わない(DroidInputs は
exterior_image_1_left と wrist_image_left だけを使う)ので、黒画像を入れる(書き出しを軽くするため)。
最後に、状態と行動の分位点を DROID の正規化の統計(pi05_droid の assets)と並べて表示し、JSON に保存する。

使い方: HF_LEROBOT_HOME=/tmp/lerobot uv run --no-sync python convert_lerobot.py --src /data/npz/demos --repo-id local/openarm_place
(書き出し先は手元のディスクにする。Modal の Volume に小さな PNG を大量に書くと、1 エピソード 1 分かかった)
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        shutil.rmtree(out)
    img = {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]}
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, robot_type="panda", fps=15,
        features={"exterior_image_1_left": img, "exterior_image_2_left": img, "wrist_image_left": img,
                  "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
                  "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
                  "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]}},
        image_writer_threads=10, image_writer_processes=5)

    files = sorted(glob.glob(os.path.join(args.src, "ep_*.npz")))
    black = np.zeros((180, 320, 3), np.uint8)
    states, actions, prompts = [], [], {}
    for i, f in enumerate(files):
        d = np.load(f)
        prompt = str(d["prompt"])
        prompts[prompt] = prompts.get(prompt, 0) + 1
        for t in range(len(d["actions"])):
            ds.add_frame({"exterior_image_1_left": d["exterior_image_1_left"][t],
                          "exterior_image_2_left": black,
                          "wrist_image_left": d["wrist_image_left"][t],
                          "joint_position": d["joint_position"][t].astype(np.float32),
                          "gripper_position": d["gripper_position"][t].astype(np.float32),
                          "actions": d["actions"][t].astype(np.float32),
                          "task": prompt})
        ds.save_episode()
        states.append(np.concatenate([d["joint_position"], d["gripper_position"]], 1))
        actions.append(d["actions"])
        print(f"[{i + 1}/{len(files)}] {os.path.basename(f)} steps {len(d['actions'])}", flush=True)

    # DROID の正規化の統計と比べる(pi05 は分位点 q01/q99 で正規化する)
    states, actions = np.concatenate(states), np.concatenate(actions)
    report = {"n_episodes": len(files), "n_frames": int(len(actions)), "prompts": prompts,
              "ours": {"state_q01": np.quantile(states, 0.01, 0).tolist(), "state_q99": np.quantile(states, 0.99, 0).tolist(),
                       "actions_q01": np.quantile(actions, 0.01, 0).tolist(), "actions_q99": np.quantile(actions, 0.99, 0).tolist()}}
    try:
        from openpi.shared import download, normalize
        stats = normalize.load(download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid/assets") / "droid")
        report["droid"] = {k: {"q01": np.asarray(v.q01).tolist(), "q99": np.asarray(v.q99).tolist()} for k, v in stats.items()}
        for key, ours in (("state", states), ("actions", actions)):
            q01, q99 = np.asarray(stats[key].q01)[:ours.shape[1]], np.asarray(stats[key].q99)[:ours.shape[1]]
            norm = (ours - q01) / np.maximum(q99 - q01, 1e-6) * 2 - 1
            frac = np.mean(np.abs(norm) > 1.5, 0)
            print(f"{key}: DROID q01 {np.round(q01, 2)}\n{key}: DROID q99 {np.round(q99, 2)}\n"
                  f"{key}: ours  q01 {np.round(np.quantile(ours, 0.01, 0), 2)}\n{key}: ours  q99 {np.round(np.quantile(ours, 0.99, 0), 2)}\n"
                  f"{key}: 正規化後に |x|>1.5 の割合 {np.round(frac, 3)}")
            report[f"{key}_frac_outside_1.5"] = frac.tolist()
    except Exception as e:  # 統計の取得に失敗しても変換は終える
        print("DROID の統計と比較できなかった:", e)
    print("prompts:", prompts)
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
