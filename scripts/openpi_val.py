"""保存された各ステップの重みについて、検証用データで損失を計算する(openpi の環境で実行)。

  loss      : 学習と同じ flow matching の損失(雑音・時刻・前処理の乱数は固定。train=False で画像の拡張なし)
  act_mse   : 雑音を固定して行動を出させ(sample_actions、10 ステップ)、教師の行動との二乗誤差
              (正規化後の空間、先頭 8 ステップ × 8 次元 = 実行する部分)
既定(--num-batches 0)では、検証データは全フレーム(33 本、端数のバッチ 1 つ分を除く)で計算する。
--split train では、学習データから検証データと同じフレーム数を、種を固定したシャッフルで選んで同じ指標を測る
(過学習の確認用。学習ログのロスは画像の拡張と乱数が入るので、直接は比べにくい)。
フレームはメモリに溜めず、チェックポイントごとに同じ種でデータの読み込みを作り直して、同じ順に読む。
結果は <ckpt_dir>/metrics_<split>.json に追記する(同じフレーム数で計算済みのステップは飛ばす)。
(10-31〜11-01 の val.json は shuffle なしで先頭 128 フレーム = 検証の最初の 1 本だけで計算していた)

使い方: uv run --no-sync python openpi_val.py --exp-name run1 [--split val|train] [--steps 1000 2000] [--num-batches N]
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, "/root/project/scripts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--num-batches", type=int, default=0, help="0: 検証データの全フレーム分")
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

    base_cfg = openpi_cfg.make_config(args.exp_name, batch_size=args.batch_size, num_workers=4)
    cfg = openpi_cfg.with_repo(base_cfg, openpi_cfg.VAL_REPO_ID if args.split == "val" else openpi_cfg.REPO_ID)
    ckpt_dir = pathlib.Path(cfg.checkpoint_dir)
    out_path = ckpt_dir / f"metrics_{args.split}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    num_batches = args.num_batches
    if num_batches <= 0:     # 検証データの全フレーム分(drop_last なので端数のバッチは除く)
        val_cfg = openpi_cfg.with_repo(base_cfg, openpi_cfg.VAL_REPO_ID)
        num_batches = len(_data_loader.create_data_loader(val_cfg, shuffle=False, num_batches=1).torch_loader)
    n_frames = num_batches * args.batch_size
    print(f"{args.split}: {num_batches} batches x {args.batch_size} = {n_frames} frames", flush=True)

    def batches():
        # 種を固定した torch.Generator でシャッフルするので、作り直すたびに同じフレームを同じ順に返す
        return _data_loader.create_data_loader(cfg, shuffle=True, num_batches=num_batches)

    targets = {}
    if args.base:
        targets["base"] = download.maybe_download(f"{openpi_cfg.PI05_DROID}/params")
    steps = args.steps or sorted(int(p.name) for p in ckpt_dir.iterdir() if p.name.isdigit())
    for s in steps:
        targets[str(s)] = ckpt_dir / str(s) / "params"

    for key, params_dir in targets.items():
        if key in results and results[key].get("n_frames") == n_frames:
            continue
        t_start = time.time()
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
        for i, (obs, act) in enumerate(batches()):
            rng = jax.random.key(1000 + i)
            losses.append(float(jnp.mean(loss_fn(state, obs, act, rng))))
            noise = jax.random.normal(jax.random.key(2000 + i), act.shape)
            pred = sample_fn(state, obs, noise, rng)
            mses.append(float(jnp.mean(jnp.square(pred[:, :8, :8] - act[:, :8, :8]))))
        results[key] = {"loss": float(np.mean(losses)), "act_mse": float(np.mean(mses)), "n_frames": n_frames,
                        # バッチごとの値(ばらつき・信頼区間を後で出す用)
                        "loss_batches": losses, "act_mse_batches": mses}
        print(key, {k: results[key][k] for k in ("loss", "act_mse", "n_frames")}, f"{time.time() - t_start:.0f}s", flush=True)
        out_path.write_text(json.dumps(results, indent=1))
        del model, params, state, loss_fn, sample_fn


if __name__ == "__main__":
    main()
