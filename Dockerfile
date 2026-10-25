# ローカル(CPU のみ)の開発用。PPO の本学習は Kaggle の GPU で行う。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # ヘッドレス描画(ソフトウェア)。Kaggle 側では egl を使う
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl ffmpeg build-essential \
        libosmesa6 libgl1 libglew-dev libglfw3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

WORKDIR /workspace
CMD ["sleep", "infinity"]
