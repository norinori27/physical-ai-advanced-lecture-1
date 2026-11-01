"""openpi の公式の学習ループ(/openpi/scripts/train.py の main)を、openpi_cfg.make_config() の設定で回す(openpi の環境で実行)。

使い方: uv run --no-sync python openpi_train.py --exp-name run1 --steps 15000 [--smoke]
--smoke: 配線の確認用(200 ステップ、100 ステップごとに保存)。
"""
import argparse
import dataclasses
import sys

sys.path.insert(0, "/openpi/scripts")
sys.path.insert(0, "/root/project/scripts")

import openpi_cfg  # noqa: E402
import train as openpi_train  # noqa: E402  (openpi の scripts/train.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--steps", type=int, default=15_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=12, help="データ読み込みの並列数(PNG の展開が律速になりうる)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repo-id", default=None, help="学習に使う LeRobot データセット(既定は openpi_cfg.REPO_ID)")
    ap.add_argument("--init-params", default="", help="学習を始める重み(例: /ckpt/checkpoints/openarm_pi05_lora/run1/14999/params)")
    args = ap.parse_args()
    steps, save = (200, 100) if args.smoke else (args.steps, args.save_interval)
    cfg = openpi_cfg.make_config(args.exp_name, num_train_steps=steps, batch_size=args.batch_size, save_interval=save,
                                 warmup_steps=min(500, steps // 10), num_workers=args.num_workers, init_params=args.init_params,
                                 **({"repo_id": args.repo_id} if args.repo_id else {}))
    cfg = dataclasses.replace(cfg, resume=args.resume, overwrite=args.overwrite)
    print(cfg, flush=True)
    openpi_train.main(cfg)


if __name__ == "__main__":
    main()
