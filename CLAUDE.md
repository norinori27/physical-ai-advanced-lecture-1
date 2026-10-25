# CLAUDE.md

OpenArm(7 自由度アーム)の MuJoCo シミュで、RL の運搬方策と、固有受容感覚だけによる「違和感」検知を作る 1 週間のプロジェクト。講義レポート(動画必須)として提出する。

**作業を始める前に必ず [PLAN.md](PLAN.md) を読む。** 決定事項、Day 1 の結果、評価方針、今日やることはすべてそこにある。決定事項を変えたら PLAN.md も更新する。

## 環境
- ローカル: devcontainer(CPU のみ、GPU なし)。開発・小さな確認・CPU 版 MuJoCo での評価用。
- GPU 作業(PPO 学習、大規模並列)は Kaggle(T4)。ローカルからは実行できないので、ノートブックを作ってユーザーに実行してもらい、結果(JSON など)を貼ってもらう。
- 版は requirements.txt で固定済み(jax 0.10.2、mujoco / mujoco-mjx 3.17.0、warp-lang 1.18.0、brax 0.14.2、playground 0.2.0)。Kaggle では `jax[cuda12]==0.10.2`。
- モデル: `openarm_mujoco/`(git 管理外)。無ければ `git clone https://github.com/enactic/openarm_mujoco` して commit `a3c8a57` に合わせる。

## 実行方法
- スクリプトはリポジトリのルートから `python scripts/<name>.py`。`src/` は各スクリプトが `sys.path` に足している。
- 描画するときは `MUJOCO_GL=osmesa`(ローカル)。Kaggle では `egl`。
- Kaggle 用ノートブックは `python scripts/build_kaggle_notebook.py` で `src/` を埋め込んで生成する。手で .ipynb を編集しない。
- MJX の初回コンパイルは CPU で数十秒〜数分かかる。長い処理はバックグラウンドで実行し、出力を `grep` や `head` に通さない(バッファされて途中経過が見えなくなる)。

## シミュの決まりごと(理由は PLAN.md 第 6 節)
- MJX は `impl="warp"`、時間刻み 0.004 s、制御周期 0.02 s。
- シーンは `src/openarm_anomaly/scene.py` の `build_model()` で作る(台座 15 cm、指先パッドの箱、重力補償)。
- ハンドの姿勢は「真下向き」だけを拘束し、ヨーは自由。

## git
- コミットの作成者は norinori27(`--local` で設定済み)。個人情報(本名、個人のメールアドレス、旧アカウント名)をコミット・ファイルに入れない。
- push はユーザーに確認してから。

## ユーザーとのやり取り
- 日本語で話す。専門用語は意味を説明してから使う。
- 結論を先に、端的に 5 行程度で書く。詳細はその後に続ける。
- 返答は論理の流れをしっかり組む(前提 → 根拠 → 結論のつながりが追えるように)。
- 不明な部分はあいまいにしない。分からないこと・未確認のことは、そうはっきり書く。
- 評価は正直に組む。同じ信号への簡単なルールで解ける事象を、学習モデルの手柄にしない(PLAN.md 第 4 節)。
