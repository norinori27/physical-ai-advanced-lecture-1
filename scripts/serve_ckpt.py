"""学習した重み(または微調整前の pi05_droid)で openpi の推論サーバを立てる(openpi の環境で実行)。

学習と同じ openpi_cfg.make_config() で設定を作って読み込む(LoRA なしの設定で読み込むと、LoRA が黙って使われない)。
読み込み後、LoRA の重みが含まれていることを確かめ、その大きさを表示する
(openpi の LoRA は A・B とも標準偏差 0.01 の正規分布で初期化されるので、学習前でも 0 ではない。平均絶対値は約 0.008)。

使い方: uv run --no-sync python serve_ckpt.py --exp-name run1 --step 3000 [--port 8000]
        uv run --no-sync python serve_ckpt.py --base   # 微調整前(openpi 公式の pi05_droid 設定)
"""
import argparse
import pathlib
import sys

sys.path.insert(0, "/root/project/scripts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name")
    ap.add_argument("--step", type=int)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import jax
    import numpy as np
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    if args.base:
        policy = policy_config.create_trained_policy(_config.get_config("pi05_droid"), "gs://openpi-assets/checkpoints/pi05_droid")
    else:
        import openpi_cfg
        cfg = openpi_cfg.make_config(args.exp_name)
        ckpt = pathlib.Path(cfg.checkpoint_dir) / str(args.step)
        policy = policy_config.create_trained_policy(cfg, ckpt)
        from flax import nnx
        flat = jax.tree_util.tree_leaves_with_path(nnx.state(policy._model).to_pure_dict())
        lora = [float(np.abs(np.asarray(v, np.float32)).mean()) for p, v in flat
                if "lora_b" in jax.tree_util.keystr(p)]
        mean = float(np.mean(lora)) if lora else float("nan")
        print(f"LoRA B の重み: {len(lora)} 個、平均絶対値 {mean:.3e}(初期値なら約 0.008)", flush=True)
        if not lora:
            raise RuntimeError("LoRA の重みが見つからない(LoRA なしの設定で読み込んでいる)")
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port,
                                                           metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
