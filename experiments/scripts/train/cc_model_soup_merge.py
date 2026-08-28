#!/usr/bin/env python3
"""Model-soup (linear interpolation) merge of an SFT model with its base.

    merged[key] = alpha * sft[key] + (1 - alpha) * base[key]      (float tensors)
                = sft[key]                                          (non-float: copied)

!! OUTPUT DTYPE MATTERS !!  (2026-08-26, found by the taste-eval soup autopsy)

bf16 carries 8 mantissa bits, i.e. one ulp is a 0.39% relative step.  A 1-epoch
full-FT at lr 5e-6 moves the median weight by only ~1.3 bf16 ulp, so at alpha=0.1
the intended step is ~0.13 ulp: round-to-nearest sends 86% of weights straight
back to the base value and snaps the survivors to a whole ulp.  The bf16 soups
built before this fix measured alpha_eff = 0.023 (not 0.10) with cosine 0.42 to
the intended direction -- i.e. mostly base plus a sparse quantisation kick, which
made them terminate WORSE than either parent on long generations.

Measured fidelity on Qwen3.5-4B / full_wd01 at alpha=0.1:
    bf16   alpha_eff 0.022   ||got||/||want|| 0.55     <- unusable
    fp16   alpha_eff 0.115   ||got||/||want|| 1.16     <- fine
    fp32   alpha_eff 0.100   ||got||/||want|| 1.00     <- exact

So: for alpha below ~0.4, pass --out-dtype float16 (or float32).  The script now
VERIFIES alpha_eff after writing and fails if it drifted.

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
    # Lazily open each shard ONCE and keep the safe_open handle (mmap). Do NOT cache
    # whole tensors — the old version cached f.get_tensor() for every key in a shard,
    # pinning ~36GB/model in RAM and making the merge I/O-bound + OOM-prone. With mmap
    # the OS page cache handles reuse; we only materialize the one tensor we need.
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
    ap.add_argument("--device", default="cpu", help="'cuda' does the per-tensor float "
                    "arithmetic on GPU (much faster); tensors still saved from CPU.")
    ap.add_argument("--out-dtype", default="auto",
                    choices=("auto", "keep", "float16", "bfloat16", "float32"),
                    help="auto = keep the SFT dtype when alpha >= 0.4, else float16. "
                         "bf16 CANNOT represent a small-alpha soup (see module docstring).")
    ap.add_argument("--alpha-tol", type=float, default=0.25,
                    help="fail if the realised alpha_eff differs from --alpha by more than "
                         "this relative amount. fp16 round-to-nearest lands ~15%% high at "
                         "alpha=0.1 (measured), which is fine; bf16 lands ~75%% LOW, which "
                         "is not -- that is the gap this guard exists to catch.")
    ap.add_argument("--no-verify", action="store_true", help="skip the post-write alpha check")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("WARN --device cuda requested but no CUDA available; falling back to cpu")
        dev = "cpu"
    assert 0.0 <= args.alpha <= 1.0

    DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    _DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if args.out_dtype == "keep":
        out_dtype = None
    elif args.out_dtype == "auto":
        out_dtype = None if args.alpha >= 0.4 else torch.float16
        if out_dtype is not None:
            print(f"[dtype] alpha={args.alpha} < 0.4 -> writing float16 "
                  f"(bf16 would round most of the step away)")
    else:
        out_dtype = _DT[args.out_dtype]

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
            merged = t_sft.to(out_dtype).contiguous() if (
                out_dtype is not None and torch.is_floating_point(t_sft)) else t_sft.contiguous()
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
            odt = out_dtype if out_dtype is not None else t_sft.dtype
            merged = (t_sft.to(dev).float().mul(a).add_(t_base.to(dev).float(), alpha=1.0 - a)
                      ).to(odt).cpu().contiguous()
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
    # the written config still advertises the SFT dtype; make it match what we wrote
    if out_dtype is not None:
        name = str(out_dtype).replace("torch.", "")
        cfgp = os.path.join(args.out, "config.json")
        if os.path.exists(cfgp):
            cfg = json.load(open(cfgp))
            for fld in ("dtype", "torch_dtype"):
                if fld in cfg:
                    cfg[fld] = name
            if isinstance(cfg.get("text_config"), dict):
                for fld in ("dtype", "torch_dtype"):
                    if fld in cfg["text_config"]:
                        cfg["text_config"][fld] = name
            json.dump(cfg, open(cfgp, "w"), indent=2)
            print(f"[dtype] config.json dtype -> {name}")

    print(f"alpha={a}: mixed {nmix} float tensors, copied {ncopy} non-float; "
          f"{len(shard_names) or 1} shard(s), {total/1e9:.2f} GB -> {args.out}")

    if not args.no_verify:
        verify_alpha(args.out, args.base, args.sft, a, args.alpha_tol)


def verify_alpha(out_dir, base_dir, sft_dir, alpha, tol, n_probe=12):
    """Re-read what we wrote and measure the alpha the WEIGHTS actually encode.

    alpha_eff = <merged - base, sft - base> / ||sft - base||^2, which is exactly
    `alpha` for a faithful merge and collapses toward 0 when the output dtype is
    too coarse to hold the step.  Also reports how many weights came out bitwise
    identical to base -- the tell-tale of a rounded-away soup.
    """
    km_o, km_b, km_s = load_keymap(out_dir), load_keymap(base_dir), load_keymap(sft_dir)
    co, cb, cs = {}, {}, {}
    keys = [k for k in sorted(set(km_o) & set(km_b) & set(km_s))
            if k.endswith(("mlp.up_proj.weight", "mlp.down_proj.weight",
                           "self_attn.q_proj.weight", "self_attn.o_proj.weight"))]
    if not keys:
        print("[verify] no probe tensors found; skipping")
        return
    keys = keys[:: max(1, len(keys) // n_probe)][:n_probe]
    num = den = gn = wn = same = tot = 0.0
    for k in keys:
        tb = get_tensor(km_b, cb, k).float()
        ts = get_tensor(km_s, cs, k).float()
        to_ = get_tensor(km_o, co, k)
        same += (to_.float() == tb).sum().item()
        tot += tb.numel()
        d = ts - tb
        got = to_.float() - tb
        want = alpha * d
        num += (got * d).sum().item(); den += (d * d).sum().item()
        gn += (got * got).sum().item(); wn += (want * want).sum().item()
    a_eff = num / max(den, 1e-30)
    ratio = (gn ** 0.5) / max(wn ** 0.5, 1e-30)
    print(f"[verify] probed {len(keys)} tensors: alpha_eff={a_eff:.4f} (asked {alpha}), "
          f"||realised||/||intended||={ratio:.2f}, bitwise-identical-to-base={same/tot:.1%}")
    if alpha > 0 and abs(a_eff - alpha) / alpha > tol:
        raise SystemExit(
            f"[verify] FAILED: alpha_eff={a_eff:.4f} is off by "
            f"{abs(a_eff-alpha)/alpha:.0%} (tol {tol:.0%}). The output dtype cannot hold "
            f"this step -- rerun with --out-dtype float16 (or float32).")


if __name__ == "__main__":
    main()
