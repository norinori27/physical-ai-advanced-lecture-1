"""描画用に、見た目のメッシュを間引く。

見た目用メッシュは全体で約 50 万頂点あり、CPU の描画(osmesa)は画素数に関係なく 1 フレーム約 0.33 s かかる。
VLA の入力は 224x224 程度なので細部は要らない。間引いた頂点・面をキャッシュし、MjSpec のメッシュへ直接入れる。
色は XML のマテリアルで決まるので、形状だけの置き換えで済む。物理には関わらない(衝突は無い)。
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "light_meshes.npz"
MESHDIR = ROOT / "openarm_mujoco" / "v1" / "meshes"


def _build_cache(spec, target_faces):
    import fast_simplification
    import trimesh

    out = {}
    for mesh in spec.meshes:
        if mesh.name.endswith("_collision"):
            continue
        path = MESHDIR / mesh.file
        tm = trimesh.load(path, force="mesh", process=False)
        v, f = np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int32)
        if len(f) > target_faces:
            ratio = 1.0 - target_faces / len(f)
            v, f = fast_simplification.simplify(v.astype(np.float32), f, target_reduction=ratio)
        out[mesh.name + "__v"], out[mesh.name + "__f"] = np.asarray(v, np.float64), np.asarray(f, np.int32)
    CACHE.parent.mkdir(exist_ok=True)
    # 一時ファイルに書いてから置き換える(複数のプロセスが同時に作っても、書きかけを読まない)
    tmp = CACHE.with_name(f"{CACHE.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, CACHE)


def lighten_visual_meshes(spec, target_faces=1500):
    """spec の見た目用メッシュを、面数 target_faces 程度に間引いたものへ置き換える。"""
    if not CACHE.exists():
        _build_cache(spec, target_faces)
    cache = np.load(CACHE)
    for mesh in spec.meshes:
        if mesh.name + "__v" in cache:
            mesh.file = ""
            mesh.uservert = cache[mesh.name + "__v"].ravel()
            mesh.userface = cache[mesh.name + "__f"].ravel()
    return spec
