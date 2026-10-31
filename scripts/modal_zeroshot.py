"""Modal の GPU で pi05_droid をゼロショットで動かし、mp4 と JSON を手元に保存する。

1 つのコンテナの中で、openpi の推論サーバ(公式の scripts/serve_policy.py --env=DROID)をバックグラウンドで起動し、
scripts/run_vla.py をそのサーバ(localhost)につないでシミュを回す。
openpi(numpy<2、JAX)とシミュ側(requirements.txt)は、別の Python 環境に分けている。

使い方(ホストで、リポジトリのルートから。openarm_mujoco/ を先に取得しておく):
    modal run scripts/modal_zeroshot.py --prompt "pick up the orange cube" --steps 150 --episodes 5 --name zs_pick
仮想 Franka 経由の付け替え(条件 B)は --retarget を付ける(例: --retarget --name zs_rt_pick)。
出力: outputs/zeroshot/<name>_ep<i>.mp4 と .json(i = 0, 1, ...)
openpi の Policy は乱数の種を指定しないと jax.random.key(0) から始め、推論ごとに分割して進める(policy.py)。
そのため、同じサーバに続けて流したエピソードは別のノイズになり、新しいコンテナでは同じ列が繰り返される。
初回はイメージのビルドと重みのダウンロード(Volume "openpi-cache" に保存)で時間がかかる。
"""
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"   # 2026-08-24 の main
PROJECT = "/root/project"
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "ffmpeg", "build-essential",
                 "libosmesa6", "libgl1", "libegl1", "libglew-dev", "libglfw3")
    # openpi: 公式リポジトリを特定のコミットで取得し、同梱の uv.lock どおりに入れる(README の手順)。
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/Physical-Intelligence/openpi.git /openpi",
        f"cd /openpi && git checkout {OPENPI_COMMIT} && git submodule update --init --recursive",
        "cd /openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen",
    )
    # シミュ側: 版とハッシュを固定した requirements.txt と、openpi 同梱の openpi-client。
    .add_local_file(ROOT / "requirements.txt", "/tmp/requirements.txt", copy=True)
    .run_commands(
        "pip install --require-hashes -r /tmp/requirements.txt",
        "pip install /openpi/packages/openpi-client",
    )
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "OPENPI_DATA_HOME": CACHE})
    .add_local_dir(ROOT / "src", f"{PROJECT}/src")
    .add_local_dir(ROOT / "scripts", f"{PROJECT}/scripts")
    .add_local_dir(ROOT / "openarm_mujoco" / "v1", f"{PROJECT}/openarm_mujoco/v1")
)

app = modal.App("openarm-vla-zeroshot", image=image)
cache = modal.Volume.from_name("openpi-cache", create_if_missing=True)


# L4 が空かないときは、順に L40S、A100 40GB を試す(推論は 8 GB 超で足りる)。
@app.function(gpu=["L4", "L40S", "A100-40GB"], timeout=3600, volumes={CACHE: cache})
def run(prompt: str, steps: int, horizon: int, delta_scale: float, episodes: int,
        retarget: bool = False, track_yaw: bool = True, hide_fingers: bool = False) -> list[dict]:
    import subprocess
    import time

    from websockets.sync.client import connect

    server = subprocess.Popen(["uv", "run", "--no-sync", "scripts/serve_policy.py", "--env=DROID"], cwd="/openpi")
    # 初回は重みのダウンロードがあるので長めに待つ。
    t0 = time.time()
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"推論サーバが終了した(code {server.returncode})")
        try:
            # 正規の WebSocket 接続で確かめる(素の TCP 接続だと、サーバが handshake 失敗のログを出す)。
            connect("ws://localhost:8000", open_timeout=2, max_size=None).close()
            break
        except OSError:
            if time.time() - t0 > 1800:
                raise TimeoutError("推論サーバが 30 分以内に起動しなかった")
            time.sleep(5)
    print(f"server ready in {time.time() - t0:.0f}s")
    cache.commit()

    extra = ((["--retarget"] if retarget else []) + ([] if track_yaw else ["--no-track-yaw"])
             + (["--hide-fingers"] if hide_fingers else []))
    results = []
    try:
        for i in range(episodes):
            out = f"/tmp/out_ep{i}.mp4"
            subprocess.run(["python", "scripts/run_vla.py", "--host", "localhost", "--port", "8000",
                            "--prompt", prompt, "--steps", str(steps), "--horizon", str(horizon),
                            "--delta-scale", str(delta_scale), "--seed", str(i), "--out", out, *extra],
                           cwd=PROJECT, check=True)
            results.append({"mp4": pathlib.Path(out).read_bytes(),
                            "json": pathlib.Path(out.replace(".mp4", ".json")).read_bytes()})
    finally:
        server.terminate()
    return results


@app.local_entrypoint()
def main(prompt: str = "pick up the orange cube", steps: int = 150, horizon: int = 8,
         delta_scale: float = 0.2, episodes: int = 1, name: str = "zs",
         retarget: bool = False, track_yaw: bool = True, hide_fingers: bool = False):
    results = run.remote(prompt, steps, horizon, delta_scale, episodes, retarget, track_yaw, hide_fingers)
    out_dir = ROOT / "outputs" / "zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, res in enumerate(results):
        for ext in ("mp4", "json"):
            (out_dir / f"{name}_ep{i}.{ext}").write_bytes(res[ext])
            print("saved", out_dir / f"{name}_ep{i}.{ext}")
