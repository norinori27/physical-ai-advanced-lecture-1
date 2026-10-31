"""Modal で、デモの変換 → LoRA 微調整 → 検証ロス → シミュ評価を行う。学習中でも、保存済みの重みで検証・評価を並行できる。

Volume: openpi-cache(重みなどの取得物、/cache)、openarm-data(デモと LeRobot データ、/data)、openarm-ckpt(重み・ログ・評価、/ckpt)

使い方(ホストで、リポジトリのルートから):
  1. modal run scripts/modal_train.py --action upload                   # data/demos, data/demos_val を Volume へ
  2. modal run scripts/modal_train.py --action convert                  # LeRobot 形式へ変換。DROID の統計との比較を表示
  3. modal run scripts/modal_train.py --action smoke                    # 200 ステップの配線確認(学習 → 保存 → 検証 → 評価 1 本)
  4. modal run --detach scripts/modal_train.py --action train --exp run1 --steps 15000   # 本番(端末を閉じても続く)
  5. modal run scripts/modal_train.py --action val --exp run1           # 保存済みの全ステップの検証ロス(計算済みは飛ばす)
     modal run scripts/modal_train.py --action eval --exp run1 --step 3000 --seeds 5      # 途中の重みでシミュ評価
     modal run scripts/modal_train.py --action eval_base --seeds 5      # 微調整前(pi05_droid)を同じ条件で評価
  6. modal run scripts/modal_train.py --action analyze --exp run1 --step 8000 --tag run1_s8000      # 考察用の分析
     modal run scripts/modal_train.py --action analyze_base --tag base                             # 微調整前で同じ分析
  7. modal run scripts/modal_train.py --action fetch --tag run1_s3000   # 評価・分析の結果(jsonl、mp4、json)を outputs/eval/ へ
     重い npz は `modal volume get openarm-ckpt eval/<tag> outputs/eval/` で取得する。
  学習のログ: `modal volume get openarm-ckpt logs/run1.log -`
"""
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"   # 2026-08-24 の main(modal_zeroshot.py と同じ)
PROJECT = "/root/project"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "ffmpeg", "build-essential",
                 "libosmesa6", "libgl1", "libegl1", "libglew-dev", "libglfw3")
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/Physical-Intelligence/openpi.git /openpi",
        f"cd /openpi && git checkout {OPENPI_COMMIT} && git submodule update --init --recursive",
        "cd /openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen",
    )
    .add_local_file(ROOT / "requirements.txt", "/tmp/requirements.txt", copy=True)
    .run_commands(
        "pip install --require-hashes -r /tmp/requirements.txt",
        "pip install /openpi/packages/openpi-client",
    )
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "OPENPI_DATA_HOME": "/cache",
          "HF_LEROBOT_HOME": "/tmp/lerobot", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9"})
    .add_local_dir(ROOT / "src", f"{PROJECT}/src")
    .add_local_dir(ROOT / "scripts", f"{PROJECT}/scripts")
    .add_local_dir(ROOT / "openarm_mujoco" / "v1", f"{PROJECT}/openarm_mujoco/v1")
)

app = modal.App("openarm-vla-train", image=image)
cache = modal.Volume.from_name("openpi-cache", create_if_missing=True)
data = modal.Volume.from_name("openarm-data", create_if_missing=True)
ckpt = modal.Volume.from_name("openarm-ckpt", create_if_missing=True)
VOLS = {"/cache": cache, "/data": data, "/ckpt": ckpt}
UV = ["uv", "run", "--no-sync", "--project", "/openpi", "python"]


LEROBOT_TARS = ("/data/lerobot_tar/openarm_place.tar", "/data/lerobot_tar/openarm_place_val.tar")


def _extract_datasets():
    """LeRobot のデータセット(Volume 上の tar)を手元のディスク(HF_LEROBOT_HOME=/tmp/lerobot)に展開する。
    小さな PNG を Volume から直接読むと遅いため。"""
    import os
    import tarfile
    import time
    t0 = time.time()
    os.makedirs("/tmp/lerobot/local", exist_ok=True)
    for tar in LEROBOT_TARS:
        if not os.path.exists(tar):     # 変換がまだのものは飛ばす(smoke は検証用データだけで回せる)
            print("skip (not converted yet):", tar, flush=True)
            continue
        with tarfile.open(tar) as tf:
            tf.extractall("/tmp/lerobot/local")
    print(f"datasets extracted in {time.time() - t0:.0f}s", flush=True)


def _run(cmd, log=None, cwd=PROJECT):
    """サブプロセスを実行し、出力を表示しつつ(必要なら)ログファイルにも書く。"""
    import subprocess
    import sys
    f = open(log, "a") if log else None
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        if f:
            f.write(line)
            f.flush()
    p.wait()
    if f:
        f.close()
    if p.returncode != 0:
        raise RuntimeError(f"失敗: {' '.join(cmd)} (code {p.returncode})")


@app.function(volumes=VOLS, cpu=16, memory=32768, timeout=4 * 3600)
def convert(which: str = "both"):
    """手元のディスク(/tmp/lerobot)に変換し、1 つの tar にまとめて Volume に置く。which: both / train / val"""
    import os
    import tarfile
    os.makedirs("/data/lerobot_tar", exist_ok=True)
    jobs = (("/data/npz/demos", "local/openarm_place", "train", LEROBOT_TARS[0]),
            ("/data/npz/demos_val", "local/openarm_place_val", "val", LEROBOT_TARS[1]))
    for src, repo, name, tar in jobs:
        if which != "both" and which != name:
            continue
        _run(UV + [f"{PROJECT}/scripts/convert_lerobot.py", "--src", src, "--repo-id", repo,
                   "--stats-out", f"/data/stats_{name}.json"])
        with tarfile.open(tar + ".tmp", "w") as tf:
            tf.add(f"/tmp/lerobot/{repo}", arcname=repo.split("/")[1])
        os.replace(tar + ".tmp", tar)
        print("saved", tar, f"{os.path.getsize(tar) / 1e9:.2f} GB", flush=True)
    cache.commit()
    data.commit()


@app.function(volumes=VOLS, gpu=["H100", "A100-80GB"], cpu=8, memory=65536, timeout=24 * 3600)
def train(exp: str, steps: int, smoke: bool = False, resume: bool = False, overwrite: bool = False, repo: str = ""):
    import os
    os.makedirs("/ckpt/logs", exist_ok=True)
    _extract_datasets()
    cmd = UV + [f"{PROJECT}/scripts/openpi_train.py", "--exp-name", exp, "--steps", str(steps)]
    cmd += ["--smoke"] if smoke else []
    cmd += ["--resume"] if resume else []
    cmd += ["--overwrite"] if overwrite else []
    cmd += ["--repo-id", repo] if repo else []
    _run(cmd, log=f"/ckpt/logs/{exp}.log")
    ckpt.commit()


@app.function(volumes=VOLS, gpu=["L40S", "A100-40GB", "A100-80GB"], timeout=3 * 3600)
def val(exp: str, steps: str = "", base: bool = False):
    _extract_datasets()
    cmd = UV + [f"{PROJECT}/scripts/openpi_val.py", "--exp-name", exp]
    cmd += ["--steps", *steps.split(",")] if steps else []
    cmd += ["--base"] if base else []
    _run(cmd, log=f"/ckpt/logs/{exp}_val.log")
    ckpt.commit()
    import json
    return json.loads(pathlib.Path(f"/ckpt/checkpoints/openarm_pi05_lora/{exp}/val.json").read_text())


@app.function(volumes=VOLS, gpu=["L4", "L40S", "A100-40GB"], cpu=4, timeout=6 * 3600)
def evaluate(tag: str, exp: str = "", step: int = 0, base: bool = False, conditions: str = "", seeds: int = 5,
             max_steps: int = 450):
    import subprocess
    import time

    import numpy as np
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect

    serve = UV + [f"{PROJECT}/scripts/serve_ckpt.py"] + (["--base"] if base else ["--exp-name", exp, "--step", str(step)])
    server = subprocess.Popen(serve, cwd=PROJECT)
    t0 = time.time()
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"推論サーバが終了した(code {server.returncode})")
        try:
            connect("ws://localhost:8000", open_timeout=2, max_size=None).close()
            break
        except OSError:
            if time.time() - t0 > 1800:
                raise TimeoutError("推論サーバが 30 分以内に起動しなかった")
            time.sleep(5)
    cache.commit()
    # 初回推論の JAX コンパイル中に keepalive で切断されないよう、ping を切った接続で 1 回推論しておく
    with connect("ws://localhost:8000", max_size=None, compression=None, ping_interval=None) as ws:
        ws.recv()
        img = np.zeros((224, 224, 3), np.uint8)
        ws.send(msgpack_numpy.Packer().pack({
            "observation/exterior_image_1_left": img, "observation/wrist_image_left": img,
            "observation/joint_position": np.zeros(7), "observation/gripper_position": np.zeros(1), "prompt": "warm up"}))
        ws.recv()
    print(f"server ready + warm-up in {time.time() - t0:.0f}s", flush=True)
    try:
        cmd = ["python", "scripts/eval_vla.py", "--tag", tag, "--seeds", str(seeds), "--max-steps", str(max_steps),
               "--out", "/ckpt/eval"]
        cmd += ["--conditions", *conditions.split(",")] if conditions else []
        _run(cmd)
    finally:
        server.terminate()
        ckpt.commit()


@app.function(volumes=VOLS, gpu=["L40S", "A100-40GB", "A100-80GB"], timeout=4 * 3600)
def analyze(tag: str, exp: str = "", step: int = 0, base: bool = False, episodes_tag: str = "", what: str = "",
            max_episodes: int = 6):
    cmd = UV + [f"{PROJECT}/scripts/analyze_policy.py", "--episodes", f"/ckpt/eval/{episodes_tag or tag}",
                "--out", f"/ckpt/analysis/{tag}", "--max-episodes", str(max_episodes)]
    cmd += ["--base"] if base else ["--exp-name", exp, "--step", str(step)]
    cmd += ["--what", *what.split(",")] if what else []
    _run(cmd, log=f"/ckpt/logs/analysis_{tag}.log")
    ckpt.commit()


def _exists(path):
    try:
        next(iter(ckpt.listdir(path)), None)
        return True
    except Exception:
        return False


@app.local_entrypoint()
def main(action: str, exp: str = "run1", steps: int = 15_000, step: int = 0, seeds: int = 5, conditions: str = "",
         tag: str = "", val_steps: str = "", resume: bool = False, overwrite: bool = False, max_steps: int = 450,
         episodes_tag: str = "", what: str = ""):
    if action == "upload":
        with data.batch_upload(force=True) as b:
            b.put_directory(str(ROOT / "data" / "demos"), "/npz/demos")
            b.put_directory(str(ROOT / "data" / "demos_val"), "/npz/demos_val")
        print("uploaded")
    elif action == "convert":
        convert.remote()
    elif action == "convert_val":     # 検証用だけ(smoke を早く始めるため)
        convert.remote("val")
    elif action == "convert_train":
        convert.remote("train")
    elif action == "smoke":
        name = f"{exp}_smoke"
        # 配線の確認なので、学習データの変換を待たずに検証用データで学習する
        train.remote(name, 200, smoke=True, overwrite=True, repo="local/openarm_place_val")
        print(val.remote(name, base=True))
        evaluate.remote(f"{name}_s199", exp=name, step=199, conditions="base", seeds=1, max_steps=60)
        print("smoke OK: 学習 → 保存 → 検証 → 推論サーバ → 評価 まで通った")
    elif action == "smoke_rest":     # smoke の学習が済んでいるとき、検証と評価だけを確かめる
        name = f"{exp}_smoke"
        print(val.remote(name, base=True))
        evaluate.remote(f"{name}_s199", exp=name, step=199, conditions="base", seeds=1, max_steps=60)
        print("smoke OK: 検証 → 推論サーバ → 評価 まで通った")
    elif action == "train":
        call = train.spawn(exp, steps, resume=resume, overwrite=overwrite)
        print("spawned:", call.object_id, "(--detach で実行していれば、端末を閉じても続く)")
    elif action == "val":
        print(val.remote(exp, steps=val_steps, base=True))
    elif action == "eval":
        evaluate.remote(tag or f"{exp}_s{step}", exp=exp, step=step, conditions=conditions, seeds=seeds, max_steps=max_steps)
    elif action == "eval_base":
        evaluate.remote(tag or "base", base=True, conditions=conditions, seeds=seeds, max_steps=max_steps)
    elif action == "analyze":     # 例: --action analyze --exp run1 --step 8000 --tag run1_s8000 [--episodes-tag base]
        analyze.remote(tag or f"{exp}_s{step}", exp=exp, step=step, episodes_tag=episodes_tag, what=what)
    elif action == "analyze_base":
        analyze.remote(tag or "base", base=True, episodes_tag=episodes_tag, what=what)
    elif action == "fetch":
        out = ROOT / "outputs" / "eval" / tag
        out.mkdir(parents=True, exist_ok=True)
        entries = list(ckpt.listdir(f"/eval/{tag}")) if _exists(f"/eval/{tag}") else []
        entries += list(ckpt.listdir(f"/analysis/{tag}")) if _exists(f"/analysis/{tag}") else []
        for entry in entries:
            name = pathlib.Path(entry.path).name
            if name.endswith((".jsonl", ".mp4", ".json")):
                with open(out / name, "wb") as f:
                    for chunk in ckpt.read_file(entry.path):
                        f.write(chunk)
        print("fetched to", out)
    else:
        raise ValueError(action)
