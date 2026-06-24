#!/usr/bin/env python3
"""Model-soup (linear interpolation) merge of an SFT model with its base.

    merged[key] = alpha * sft[key] + (1 - alpha) * base[key]      (float tensors)
                = sft[key]                                          (non-float: copied)

Robust to different shard layouts: matches tensors BY KEY (loads each model's full
state dict by key, regardless of how many .safetensors shards each uses). Copies
config/tokenizer from the SFT model (so chat template / special tokens are preserved).

Usage:
  python cc_model_soup_merge.py --sft <sft_dir> --base <base_dir> --alpha 0.9 --out <out_dir>
"""
from __future__ import annotations
import argparse, glob, json, os, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_keymap(model_dir: str) -> dict[str, str]:
    """key -> shard file path, supporting single- or multi-shard safetensors."""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in wm.items()}
    shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
    assert shards, f"no safetensors in {model_dir}"
    keymap = {}
    for sp in shards:
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                keymap[k] = sp
    return keymap


def get_tensor(keymap, cache, key):
    sp = keymap[key]
    if sp not in cache:
        cache[sp] = safe_open(sp, framework="pt", device="cpu")
    return cache[sp].get_tensor(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--alpha", type=float, default=0.9, help="weight on the SFT model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-size-gb", type=float, default=9.0)
    args = ap.parse_args()
    assert 0.0 <= args.alpha <= 1.0

    sft_map = load_keymap(args.sft)
    base_map = load_keymap(args.base)
    sk, bk = set(sft_map), set(base_map)
    # NOTE: full-FT in LLaMA-Factory drops the Qwen3.5 multi-token-prediction head
    # (mtp.*), so the SFT model has FEWER keys than its start/base model. That is a
    # benign architectural difference, NOT a corruption -- do not fail on it. The
    # merged model takes EXACTLY the SFT model's key set (config/tokenizer come from
    # the SFT model below, so the result is architecturally consistent = no MTP):
    #   * key in both           -> alpha*sft + (1-alpha)*base   (the soup)
    #   * key only in sft        -> copy sft  (kept; should be empty in practice)
    #   * key only in base (mtp) -> DROPPED   (not in output; matches SFT config)
    if sk != bk:
        only_sft, only_base = sorted(sk - bk), sorted(bk - sk)
        print(f"WARN key asymmetry (benign, e.g. MTP head): {len(only_sft)} only-in-sft "
              f"(copied), {len(only_base)} only-in-base (dropped)\n"
              f"  sft-only sample: {only_sft[:3]}\n  base-only sample: {only_base[:3]}")

    os.makedirs(args.out, exist_ok=True)
    # config/tokenizer/etc. from the SFT model (skip weights/index)
    for f in sorted(glob.glob(os.path.join(args.sft, "*"))):
        b = os.path.basename(f)
        if b.endswith(".safetensors") or b.endswith(".safetensors.index.json"):
            continue
        if os.path.isfile(f):
            shutil.copy2(f, os.path.join(args.out, b))

    sft_cache, base_cache = {}, {}
    keys = sorted(sft_map)
    shard_limit = int(args.shard_size_gb * 1e9)
    weight_map, cur, cur_bytes, shard_idx, total = {}, {}, 0, 1, 0
    nmix = ncopy = 0
    shard_names = []

    def flush():
        nonlocal cur, cur_bytes, shard_idx
        if not cur:
            return
        name = f"model-{shard_idx:05d}.safetensors"
        save_file(cur, os.path.join(args.out, name), metadata={"format": "pt"})
        for k in cur:
            weight_map[k] = name
        shard_names.append(name)
        cur, cur_bytes = {}, 0
        shard_idx += 1

    a = args.alpha
    for key in keys:
        t_sft = get_tensor(sft_map, sft_cache, key)
        # key only in sft (no base counterpart) -> copy sft through, no mixing
        if key not in base_map:
            merged = t_sft.contiguous()
            ncopy += 1
            cur[key] = merged
            nbytes = merged.numel() * merged.element_size()
            cur_bytes += nbytes; total += nbytes
            if cur_bytes >= shard_limit:
                flush()
            continue
        t_base = get_tensor(base_map, base_cache, key)
        if t_sft.shape != t_base.shape:
            raise SystemExit(f"{key}: shape mismatch {tuple(t_sft.shape)} vs {tuple(t_base.shape)}")
        if torch.is_floating_point(t_sft):
            merged = (t_sft.float().mul(a).add_(t_base.float(), alpha=1.0 - a)).to(t_sft.dtype).contiguous()
            nmix += 1
        else:
            merged = t_sft.contiguous()
            ncopy += 1
        cur[key] = merged
        nbytes = merged.numel() * merged.element_size()
        cur_bytes += nbytes
        total += nbytes
        if cur_bytes >= shard_limit:
            flush()
    flush()

    # single shard -> rename to model.safetensors (HF convention)
    if len(shard_names) == 1:
        src = os.path.join(args.out, shard_names[0])
        dst = os.path.join(args.out, "model.safetensors")
        os.replace(src, dst)
        weight_map = {k: "model.safetensors" for k in weight_map}
    else:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
                  open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=2)
    print(f"alpha={a}: mixed {nmix} float tensors, copied {ncopy} non-float; "
          f"{len(shard_names) or 1} shard(s), {total/1e9:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
