"""学習した重みと元の pi05_droid の重みを、部分ごとに比べる(openpi の環境で実行。CPU で可)。

固定されているはずの部分(Gemma 2B・action expert の元の重み)は変化 0、学習される部分(SigLIP、入出力の層)は 0 でない、
LoRA は元の重みに無いので大きさだけ、を確かめる。
使い方: uv run --no-sync python param_delta.py --exp-name run1_smoke --step 199
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, "/root/project/scripts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--step", type=int, required=True)
    args = ap.parse_args()

    import jax
    import numpy as np
    from openpi.models import model as _model
    from openpi.shared import download

    import openpi_cfg

    cfg = openpi_cfg.make_config(args.exp_name)
    new = _model.restore_params(pathlib.Path(cfg.checkpoint_dir) / str(args.step) / "params", dtype=None)
    old = _model.restore_params(download.maybe_download(f"{openpi_cfg.PI05_DROID}/params"), dtype=None)
    flat_new = {jax.tree_util.keystr(p): v for p, v in jax.tree_util.tree_leaves_with_path(new)}
    flat_old = {jax.tree_util.keystr(p): v for p, v in jax.tree_util.tree_leaves_with_path(old)}

    def group(k):
        if "lora" in k:
            return "LoRA(新規)" + (" action expert" if "_1" in k else " Gemma 2B")
        if "'img'" in k:
            return "SigLIP(画像)"
        if "'llm'" in k:
            return "action expert(元の重み)" if "_1'" in k or "_1" in k.split("'")[-2] else "Gemma 2B(元の重み)"
        return "入出力の層"

    stats = {}
    for k, v in flat_new.items():
        g = group(k)
        dtype = str(np.asarray(v).dtype)
        s = stats.setdefault(g, {"n_params": 0, "n_changed_tensors": 0, "n_tensors": 0, "delta_fro2": 0.0, "w_fro2": 0.0,
                                 "dtypes": []})
        if dtype not in s["dtypes"]:
            s["dtypes"].append(dtype)
        v = np.asarray(v, np.float32)
        s["n_params"] += v.size
        s["n_tensors"] += 1
        s["w_fro2"] += float(np.sum(v ** 2))
        if k in flat_old:
            # openpi は固定した重みを bfloat16 にして保存する(scripts/train.py)。元の重みも同じ精度に丸めてから比べる
            ref = np.asarray(np.asarray(flat_old[k]).astype(np.asarray(flat_new[k]).dtype), np.float32)
            d = v - ref
            s["delta_fro2"] += float(np.sum(d ** 2))
            s["n_changed_tensors"] += int(np.any(d != 0))
    for g, s in sorted(stats.items()):
        rel = (s["delta_fro2"] / s["w_fro2"]) ** 0.5 if s["w_fro2"] else 0
        print(f"{g:26s} パラメータ {s['n_params'] / 1e6:8.1f}M  テンソル {s['n_tensors']:4d}  変化したテンソル {s['n_changed_tensors']:4d}  "
              f"||ΔW||/||W|| {rel:.2e}  ||W|| {s['w_fro2'] ** 0.5:.1f}  dtype {s['dtypes']}", flush=True)
    print("LoRA は元の重みに無いので比較しない(変化したテンソル 0 は「比較対象なし」の意味)")
    out = pathlib.Path(cfg.checkpoint_dir) / f"param_delta_{args.step}.json"
    out.write_text(json.dumps(stats, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
