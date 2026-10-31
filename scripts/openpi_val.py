"""保存された各ステップの重みについて、検証用データで損失を計算する(openpi の環境で実行)。

  loss      : 学習と同じ flow matching の損失(雑音・時刻・前処理の乱数は固定。train=False で画像の拡張なし)
  act_mse   : 雑音を固定して行動を出させ(sample_actions、10 ステップ)、教師の行動との二乗誤差
              (正規化後の空間、先頭 8 ステップ × 8 次元 = 実行する部分)
結果は <ckpt_dir>/val.json に追記する(計算済みのステップは飛ばす)。

使い方: uv run --no-sync python openpi_val.py --exp-name run1 [--steps 1000 2000] [--num-batches 8]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, "/root/project/scripts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("--num-batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--base", action="store_true", help="微調整前(pi05_droid の重み)も同じ指標で測る")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from openpi.models import model as _model
    from openpi.shared import download
    from openpi.training import data_loader as _data_loader

    import openpi_cfg

    cfg = openpi_cfg.with_repo(openpi_cfg.make_config(args.exp_name, batch_size=args.batch_size, num_workers=0),
                               openpi_cfg.VAL_REPO_ID)
    ckpt_dir = pathlib.Path(cfg.checkpoint_dir)
    out_path = ckpt_dir / "val.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    # 検証用のバッチを固定して先に作る(全ステップで同じバッチを使う)
    loader = _data_loader.create_data_loader(cfg, shuffle=False, num_batches=args.num_batches)
    batches = list(loader)
    print(f"val batches: {len(batches)} x {args.batch_size}", flush=True)

    targets = {}
    if args.base:
        targets["base"] = download.maybe_download(f"{openpi_cfg.PI05_DROID}/params")
    steps = args.steps or sorted(int(p.name) for p in ckpt_dir.iterdir() if p.name.isdigit())
    for s in steps:
        targets[str(s)] = ckpt_dir / str(s) / "params"

    for key, params_dir in targets.items():
        if key in results:
            continue
        params = _model.restore_params(params_dir, dtype=jnp.bfloat16)
        # 微調整前の重みには LoRA の項が無いので、LoRA なしの構造(公式の pi05_droid と同じ)で読み込む
        model_cfg = cfg.model if key != "base" else type(cfg.model)(pi05=True, action_dim=32, action_horizon=16)
        model = model_cfg.load(params)
        # 重みは jit の引数として渡す(クロージャで参照すると、重み全体が定数として埋め込まれて GPU のメモリが足りなくなる)。
        # openpi の nnx_utils.module_jit と同じ split / merge の形。train=False は Python の値のまま固定する。
        graphdef, state = nnx.split(model)

        @jax.jit
        def loss_fn(state, o, a, r):
            return nnx.merge(graphdef, state).compute_loss(r, o, a, train=False)

        @jax.jit
        def sample_fn(state, o, n, r):
            return nnx.merge(graphdef, state).sample_actions(r, o, noise=n)

        losses, mses = [], []
        for i, (obs, act) in enumerate(batches):
            rng = jax.random.key(1000 + i)
            losses.append(float(jnp.mean(loss_fn(state, obs, act, rng))))
            noise = jax.random.normal(jax.random.key(2000 + i), act.shape)
            pred = sample_fn(state, obs, noise, rng)
            mses.append(float(jnp.mean(jnp.square(pred[:, :8, :8] - act[:, :8, :8]))))
        results[key] = {"loss": float(np.mean(losses)), "act_mse": float(np.mean(mses))}
        print(key, results[key], flush=True)
        out_path.write_text(json.dumps(results, indent=1))
        del model, params, state, loss_fn, sample_fn


if __name__ == "__main__":
    main()
