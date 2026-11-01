# OpenArm × π0.5(VLA): ゼロショットと微調整の比較

7 自由度アーム OpenArm の MuJoCo シミュレーションで、VLA(画像と言語の指示から行動を出す大規模モデル)π0.5 を動かし、
「追加学習なし(ゼロショット)」と「LoRA 微調整後」を、スクリプト制御の教師と比べた講義レポートのコードです。

- タスク: 指示した色の箱を、指示した色のマーカーの上に置く("put the red cube on the green marker")
- モデル: [openpi](https://github.com/Physical-Intelligence/openpi) の `pi05_droid`(コミット `215abfb`)。論文: [π0.5](https://arxiv.org/abs/2504.16054)
- 結果: ゼロショット 0/30、微調整後 17/60(教師 38/60)。詳細はレポートと動画(リンクは提出物を参照)。

## 構成
| 場所 | 内容 |
|---|---|
| `src/openarm_anomaly/` | シーン(`scene.py`、`task.py`)、VLA 用の環境(`vla_env.py`)、仮想 Franka による付け替え(`franka_retarget.py`)、教師(`scripted.py`)、描画の軽量化(`light_mesh.py`) |
| `scripts/gen_demos.py` | 教師でデモを生成(DART の雑音つき) |
| `scripts/convert_lerobot.py` | デモを openpi の LeRobot 形式に変換 |
| `scripts/modal_train.py`, `openpi_cfg.py`, `openpi_train.py`, `openpi_val.py` | Modal 上の LoRA 微調整と検証 |
| `scripts/modal_zeroshot.py`, `run_vla.py` | ゼロショットの実行 |
| `scripts/eval_vla.py`, `rejudge_strict.py`, `teacher_eval.py`, `summarize_eval.py` | 評価、成功の判定、集計 |
| `scripts/make_compare_clip.py`, `make_video.py` | 提出動画の組み立て |
| `scripts/dagger*.py` | 追加実験(DAgger)。レポートの結果には含めていない |

## 実行方法
シミュ側は CPU で動きます(描画は `MUJOCO_GL=osmesa`)。GPU が要る処理(推論サーバ、微調整)は [Modal](https://modal.com) の中で動かします。

1. 環境: `docker/Dockerfile` を使うか、`requirements.txt`(版とハッシュを固定)で入れる。
2. OpenArm のモデル: [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco) をルートに `openarm_mujoco/` として置く(commit `a3c8a57`)。
3. 各スクリプトはリポジトリのルートから `python scripts/<name>.py` で実行する(`--help` で引数)。
4. 流れ: `gen_demos.py` → `modal_train.py`(upload → convert → smoke → train → eval → fetch)→ `summarize_eval.py` → `make_video.py`。

評価は 6 条件(base、pos_outer、goal_color、light、box_color、language)× 10 本、seed は固定です。
成功の定義は「置く → 指を離す → 手を 5 cm 以上上げる」(`scripts/rejudge_strict.py`)。

## 出典とライセンス
- openpi: Apache-2.0。重みは Gemma の規約([Gemma Terms of Use](https://ai.google.dev/gemma/terms))の対象と見られます。**微調整した重みは配布しません。**
- DROID: [論文](https://arxiv.org/abs/2403.12945)、データは CC-BY 4.0(このリポジトリには含めていません)。
- OpenArm のモデル: [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco)(このリポジトリには含めていません)。
- このリポジトリのコードのライセンスは未設定です(公開前に決めてください)。
