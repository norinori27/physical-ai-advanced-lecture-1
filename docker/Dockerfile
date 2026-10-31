# 第三者が解析(CPU 側のシミュと描画)を再現するためのイメージ。
# GPU 側(openpi の推論サーバ、微調整)は Modal のスクリプトの中で環境を定義する。
#
# ビルド(リポジトリのルートで):
#   docker build -f docker/Dockerfile -t openarm-vla .
# 実行(openarm_mujoco/ を先にルートへ取得しておく。手順は README.md):
#   docker run --rm -v "$PWD":/work openarm-vla python scripts/render_cameras.py
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # ヘッドレス描画(ソフトウェア)
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates ffmpeg \
        libosmesa6 libgl1 libglew-dev libglfw3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# 版とハッシュが固定済み。一致しない物が来たらビルドが失敗する。
RUN pip install --upgrade pip && pip install --require-hashes -r /tmp/requirements.txt

WORKDIR /work
