"""評価で保存した観測を方策に入れ直して、モデルが何に頼って行動を決めているかを調べる(openpi の環境、GPU で実行)。

推論サーバは通さず、方策を同じプロセスで読み込む(雑音を固定して入力だけを変えるため。サーバ経由だと推論ごとに雑音が変わる)。
行動チャンクの「意味」は、実行する先頭 8 ステップを仮想 Franka に積分した手先の変位(m)と、グリッパの指令で見る。

分析(--what で選ぶ。既定は全部):
  noise     : 雑音だけを変えて N 回推論したときの、手先の変位のばらつき(迷っている度合い)
  camera    : 外部カメラ / 手首カメラを黒にしたときの手先の変位の変化(どちらのカメラに頼るか)
  prompt    : 指示のゴールの色を入れ替えたとき / 指示を空にしたときの変化(言葉で行き先を決めているか)
  state     : 関節角に雑音を足したときの変化(画像を見ずに自分の姿勢だけで動いていないか)
  occlusion : 画像を格子状に灰色で隠したときの変化の地図(遮蔽感度マップ。画像のどこを根拠にしているか)
  attention : 行動を出す部分(action expert)のトークンから、画像の各区画と言葉のトークンへの注目度(層・雑音除去ステップで平均)
  lora      : LoRA で学習した重みの変化量 ||ΔW|| を層ごとに(重みだけで計算。--exp が必要)

使い方: uv run --no-sync python analyze_policy.py --exp-name run1 --step 8000 --episodes /ckpt/eval/run1_s8000 --out /ckpt/analysis/run1_s8000
        uv run --no-sync python analyze_policy.py --base --episodes /ckpt/eval/base --out /ckpt/analysis/base
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/root/project/scripts")
sys.path.insert(0, "/root/project/src")

ATTN = {"on": False, "store": []}


def install_attention_hook(action_horizon):
    """gemma.Attention の softmax を包み、行動トークン(suffix)からの注目度をホストに送る(jit 内でも動く jax.debug.callback)。
    方策を作る前(= jit でトレースされる前)に呼ぶこと。"""
    import jax
    from openpi.models import gemma

    orig = jax.nn.softmax

    def cb(p):
        if ATTN["on"]:
            ATTN["store"].append(np.asarray(p, np.float32))

    def softmax(x, axis=-1, **kw):
        out = orig(x, axis=axis, **kw)
        # BKGTS で、問い合わせが行動トークン(T = action_horizon)、キーがそれより長い(prefix + suffix)ときだけ
        if out.ndim == 5 and out.shape[3] == action_horizon and out.shape[4] > action_horizon:
            jax.debug.callback(cb, out.mean(axis=(1, 2, 3)))     # ヘッドと行動トークンで平均 → (B, S)
        return out

    gemma.jax.nn.softmax = softmax   # gemma.py は jax.nn.softmax を参照する(jax.nn を差し替える)


def load_policy(args):
    from openpi.policies import policy_config
    from openpi.training import config as _config
    if args.base:
        cfg = _config.get_config("pi05_droid")
        return policy_config.create_trained_policy(cfg, "gs://openpi-assets/checkpoints/pi05_droid"), cfg
    import openpi_cfg
    cfg = openpi_cfg.make_config(args.exp_name)
    return policy_config.create_trained_policy(cfg, os.path.join(cfg.checkpoint_dir, str(args.step))), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name")
    ap.add_argument("--step", type=int)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--episodes", required=True, help="eval_vla.py の出力フォルダ(npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--what", nargs="+", default=["noise", "camera", "prompt", "state", "occlusion", "attention", "lora"])
    ap.add_argument("--max-episodes", type=int, default=6, help="--per-condition が 0 のとき、ファイル名順の先頭から")
    ap.add_argument("--per-condition", type=int, default=0, help="条件ごとに seed の小さい順に何本ずつ調べるか")
    ap.add_argument("--frames", default="start,pregrasp,carry", help="各エピソードで調べる時点")
    ap.add_argument("--n-noise", type=int, default=16)
    ap.add_argument("--grid", default="6,10", help="遮蔽の格子(縦,横)")
    args = ap.parse_args()

    import jax
    from openpi_client import image_tools

    from openarm_anomaly.franka_retarget import franka_fk
    from openarm_anomaly.task import MARKERS_HELDOUT, MARKERS_TRAIN

    os.makedirs(args.out, exist_ok=True)
    tmp_cfg_h = 15 if args.base else 16
    if "attention" in args.what:
        install_attention_hook(tmp_cfg_h)
    policy, cfg = load_policy(args)
    H, A = cfg.model.action_horizon, cfg.model.action_dim

    def displacement(q, chunk):
        """先頭 8 ステップを仮想 Franka に積分した手先の変位(m)と、最後のグリッパ指令。"""
        qq = q + np.cumsum(np.clip(chunk[:8, :7], -1, 1) * 0.2, axis=0)[-1]
        return franka_fk(qq)[0] - franka_fk(q)[0], float(chunk[7, 7])

    def infer(obs, noise):
        return np.asarray(policy.infer(obs, noise=noise)["actions"])

    def make_obs(ep, t, prompt=None, ext=None, wrist=None, q=None):
        return {"observation/exterior_image_1_left": image_tools.resize_with_pad(ext if ext is not None else ep["ext"][t], 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(wrist if wrist is not None else ep["wrist"][t], 224, 224),
                "observation/joint_position": q if q is not None else ep["joint_position"][t],
                "observation/gripper_position": ep["gripper_position"][t],
                "prompt": prompt if prompt is not None else ep["task"]["prompt"]}

    def pick_frames(ep):
        """start(最初)、pregrasp(箱に最も近づいた時点)、carry(箱が最も高い時点。持ち上げていなければ省く)。"""
        tcp, box = ep["tcp"], ep["box_pos"]
        out = {"start": 0, "pregrasp": int(np.argmin(np.linalg.norm(tcp[:, :2] - box[0, :2], axis=1)))}
        if box[:, 2].max() > 0.07:
            out["carry"] = int(np.argmax(box[:, 2]))
        return {k: v for k, v in out.items() if k in args.frames.split(",")}

    results = {"tag": os.path.basename(args.out.rstrip("/")), "base": args.base, "exp": args.exp_name, "step": args.step,
               "episodes": []}
    rng = np.random.default_rng(0)
    noise0 = np.asarray(jax.random.normal(jax.random.key(0), (H, A)))
    files = sorted(glob.glob(os.path.join(args.episodes, "*.npz")))
    if args.per_condition > 0:     # ファイル名は <条件>_s<seed>.npz
        by = {}
        for f in files:
            by.setdefault(os.path.basename(f).rsplit("_s", 1)[0], []).append(f)
        files = [f for fs in by.values() for f in fs[:args.per_condition]]
    else:
        files = files[:args.max_episodes]
    print(f"{len(files)} episodes:", [os.path.basename(f) for f in files], flush=True)
    for f in files:
        d = np.load(f)
        ep = {k: d[k] for k in d.files if k != "task_json"}
        ep["task"] = json.loads(str(d["task_json"]))
        er = {"file": os.path.basename(f), "condition": ep["task"]["condition"], "prompt": ep["task"]["prompt"], "frames": {}}
        for fname, t in pick_frames(ep).items():
            q = ep["joint_position"][t]
            base_chunk = infer(make_obs(ep, t), noise0)
            d0, g0 = displacement(q, base_chunk)
            fr = {"t": t, "disp": d0.tolist(), "grip": g0, "chunk": base_chunk.tolist()}
            delta = lambda c: float(np.linalg.norm(displacement(q, c)[0] - d0))  # noqa: E731
            if "noise" in args.what:
                ds = [displacement(q, infer(make_obs(ep, t), np.asarray(jax.random.normal(jax.random.key(k + 1), (H, A)))))[0]
                      for k in range(args.n_noise)]
                fr["noise_std_m"] = float(np.linalg.norm(np.std(ds, axis=0)))
            if "camera" in args.what:
                black = np.zeros_like(ep["ext"][t])
                fr["camera"] = {"ext_black": delta(infer(make_obs(ep, t, ext=black), noise0)),
                                "wrist_black": delta(infer(make_obs(ep, t, wrist=black), noise0))}
            if "prompt" in args.what:
                task = ep["task"]
                others = [c for c in (*MARKERS_TRAIN, *MARKERS_HELDOUT) if c != task["target"]]
                fr["prompt"] = {"empty": delta(infer(make_obs(ep, t, prompt=""), noise0))}
                for c in others:
                    p = task["template"].format(box=task["box_color"], marker=c)
                    c_chunk = infer(make_obs(ep, t, prompt=p), noise0)
                    fr["prompt"][f"swap_{c}"] = delta(c_chunk)
                    fr["prompt"][f"swap_{c}_disp"] = displacement(q, c_chunk)[0].tolist()
            if "state" in args.what:
                fr["state_noise_0.1rad"] = float(np.mean([delta(infer(make_obs(ep, t, q=q + rng.normal(0, 0.1, 7)), noise0))
                                                          for _ in range(4)]))
            if "occlusion" in args.what:
                gy, gx = map(int, args.grid.split(","))
                for cam in ("ext", "wrist"):
                    img = ep[cam][t]
                    hh, ww = img.shape[0] // gy, img.shape[1] // gx
                    heat = np.zeros((gy, gx))
                    for i in range(gy):
                        for j in range(gx):
                            occ = img.copy()
                            occ[i * hh:(i + 1) * hh, j * ww:(j + 1) * ww] = 128
                            heat[i, j] = delta(infer(make_obs(ep, t, **{cam: occ}), noise0))
                    fr[f"occlusion_{cam}"] = heat.tolist()
            if "attention" in args.what:
                ATTN["store"].clear()
                ATTN["on"] = True
                infer(make_obs(ep, t), noise0)
                ATTN["on"] = False
                att = np.stack(ATTN["store"])[:, 0]          # (層 × 雑音除去ステップ, S)
                # prefix の並び: 外部画像 256 トークン(16x16)、手首画像 256、(右手首 256、マスクされている)、言葉のトークン
                fr["attention"] = {"ext_16x16": att[:, :256].mean(0).reshape(16, 16).tolist(),
                                   "wrist_16x16": att[:, 256:512].mean(0).reshape(16, 16).tolist(),
                                   "share": {"ext": float(att[:, :256].sum(1).mean()), "wrist": float(att[:, 256:512].sum(1).mean()),
                                             "text_state": float(att[:, 768:-H].sum(1).mean()),
                                             "action_tokens": float(att[:, -H:].sum(1).mean())},
                                   "n_calls": int(att.shape[0])}
            er["frames"][fname] = fr
            print(er["file"], fname, {k: v for k, v in fr.items() if k in ("noise_std_m", "camera", "state_noise_0.1rad")}, flush=True)
        results["episodes"].append(er)

    if "lora" in args.what and not args.base:
        from flax import nnx
        params = nnx.state(policy._model).to_pure_dict()
        rows = []

        def walk(node, path):
            if isinstance(node, dict):
                keys = node.keys()
                for k in keys:
                    if k.endswith("lora_a"):
                        base_name = k[:-len("_lora_a")] if k != "lora_a" else "w"
                        a, b = np.asarray(node[k], np.float32), np.asarray(node[k.replace("lora_a", "lora_b")], np.float32)
                        dw = a @ b                                   # LoRA の軸は最後の 2 軸(scaling = alpha/rank = 1)
                        w = np.asarray(node[base_name], np.float32) if base_name in node else None
                        row = {"path": "/".join(path + [base_name]), "shape": list(dw.shape), "dW_fro": float(np.linalg.norm(dw)),
                               "W_fro": float(np.linalg.norm(w)) if w is not None else None}
                        if dw.ndim >= 3:   # 層を scan でまとめている場合、先頭の軸が層
                            row["dW_per_layer"] = [float(np.linalg.norm(x)) for x in dw]
                            if w is not None and w.shape[0] == dw.shape[0]:
                                row["W_per_layer"] = [float(np.linalg.norm(x)) for x in w]
                        rows.append(row)
                for k, v in node.items():
                    walk(v, path + [str(k)])
        walk(params, [])
        results["lora"] = rows
        print(f"LoRA の層: {len(rows)}", flush=True)
    with open(os.path.join(args.out, "analysis.json"), "w") as f:
        json.dump(results, f)
    print("saved", os.path.join(args.out, "analysis.json"))


if __name__ == "__main__":
    main()
