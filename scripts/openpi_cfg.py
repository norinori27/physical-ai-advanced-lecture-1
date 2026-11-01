"""openpi の学習設定(pi05_droid から LoRA で微調整)。openpi の環境(Modal のイメージ内の /openpi)で import して使う。

公式の pi05_droid_finetune(openpi src/openpi/training/config.py)を元に、LoRA は pi0_libero_low_mem_finetune と同じ形にした。
  - 出発点: gs://openpi-assets/checkpoints/pi05_droid/params。行動は DROID と同じ関節速度 7 + グリッパ位置 1。
  - 正規化の統計: DROID のものを流用する(公式のコメント「reuse the original DROID norm stats」)。自前では計算しない。
  - LoRA: gemma_2b_lora + gemma_300m_lora。freeze_filter は Pi0Config.get_freeze_filter()、EMA なし。
  - 学習率: 最大 5e-5 のコサイン減衰(公開例の LoRA 微調整と同じ)、勾配クリップ 1.0(AdamW の既定)。
  - 重み: save_interval ごとに保存し、keep_period を同じ値にして全部残す(途中の重みで評価するため)。
推論でも必ずこの make_config() で設定を作る(LoRA なしの設定で読み込むと、エラーにならずに LoRA が使われない)。
"""
import dataclasses

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

NAME = "openarm_pi05_lora"
REPO_ID = "local/openarm_place"          # HF_LEROBOT_HOME 以下の LeRobot データセット
VAL_REPO_ID = "local/openarm_place_val"
PI05_DROID = "gs://openpi-assets/checkpoints/pi05_droid"


def model_config():
    return pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16,
                                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")


def make_config(exp_name: str, num_train_steps: int = 15_000, batch_size: int = 32, save_interval: int = 1000,
                warmup_steps: int = 500, peak_lr: float = 5e-5, repo_id: str = REPO_ID,
                checkpoint_base_dir: str = "/ckpt/checkpoints", assets_base_dir: str = "/ckpt/assets",
                num_workers: int = 4, seed: int = 42, init_params: str = "") -> _config.TrainConfig:
    """init_params: 学習を始める重み(既定は pi05_droid)。DAgger では run1 の重みから続ける(LoRA の重みも引き継がれる)。"""
    return _config.TrainConfig(
        name=NAME,
        exp_name=exp_name,
        model=model_config(),
        data=_config.LeRobotDROIDDataConfig(
            repo_id=repo_id,
            base_config=_config.DataConfig(prompt_from_task=True),
            assets=_config.AssetsConfig(assets_dir=f"{PI05_DROID}/assets", asset_id="droid"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(init_params or f"{PI05_DROID}/params"),
        freeze_filter=model_config().get_freeze_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=warmup_steps, peak_lr=peak_lr,
                                                   decay_steps=num_train_steps, decay_lr=peak_lr / 10),
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        save_interval=save_interval,
        keep_period=save_interval,
        log_interval=50,
        num_workers=num_workers,
        wandb_enabled=False,
        checkpoint_base_dir=checkpoint_base_dir,
        assets_base_dir=assets_base_dir,
        seed=seed,
    )


def with_repo(cfg: _config.TrainConfig, repo_id: str) -> _config.TrainConfig:
    """同じ設定で、読み込むデータセットだけ替える(検証用)。"""
    return dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=repo_id))
