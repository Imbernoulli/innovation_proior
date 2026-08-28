#!/usr/bin/env python3
"""Build the null model for the soup investigation.

The α=0.1 soup turns out to be, numerically, `base with a sparse ±1-ulp bf16
kick on ~11% of its weights` (the intended 0.1·(sft-base) step is smaller than
half a bf16 ulp for 86% of weights, so it rounds away; the survivors get snapped
to a whole ulp).  This script produces a model with the SAME sparsity and the
SAME ±1-ulp step size but on RANDOM positions with random signs -- i.e. all of
the quantisation damage and none of the SFT direction.

If that null degenerates like the soup does, the soup's deficit is rounding, not
model averaging.

    python make_ulp_control.py --base <base> --soup <soup> --out <dir> [--seed 0]
"""
from __future__ import annotations
import argparse, glob, json, os, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def keymap(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(d, v) for k, v in wm.items()}
    km = {}
    for sp in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                km[k] = sp
    return km


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--soup", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard-size-gb", type=float, default=9.0)
a = ap.parse_args()

kb, kp = keymap(a.base), keymap(a.soup)
os.makedirs(a.out, exist_ok=True)
for f in sorted(glob.glob(os.path.join(a.soup, "*"))):
    b = os.path.basename(f)
    if b.endswith(".safetensors") or b.endswith(".safetensors.index.json"):
        continue
    if os.path.isfile(f):
        shutil.copy2(f, os.path.join(a.out, b))

g = torch.Generator().manual_seed(a.seed)
cb, cp = {}, {}
def get(km, cache, k):
    sp = km[k]
    if sp not in cache:
        cache[sp] = safe_open(sp, framework="pt", device="cpu")
    return cache[sp].get_tensor(k)

keys = sorted(kp)
weight_map, cur, cur_bytes, shard_idx, shard_names = {}, {}, 0, 1, []
limit = int(a.shard_size_gb * 1e9)
tot_changed = tot_elems = 0

def flush():
    global cur, cur_bytes, shard_idx
    if not cur:
        return
    name = f"model-{shard_idx:05d}.safetensors"
    save_file(cur, os.path.join(a.out, name), metadata={"format": "pt"})
    for k in cur:
        weight_map[k] = name
    shard_names.append(name)
    cur, cur_bytes, shard_idx = {}, 0, shard_idx + 1

for k in keys:
    tp = get(kp, cp, k)
    if k not in kb or not torch.is_floating_point(tp):
        out = tp.contiguous()
    else:
        tb = get(kb, cb, k)
        n_changed = int((tp != tb).sum().item())
        tot_changed += n_changed
        tot_elems += tb.numel()
        if n_changed == 0 or tb.dtype != torch.bfloat16:
            out = tb.contiguous()
        else:
            bits = tb.view(torch.int16).clone().reshape(-1)
            idx = torch.randperm(bits.numel(), generator=g)[:n_changed]
            step = (torch.randint(0, 2, (n_changed,), generator=g) * 2 - 1).to(torch.int16)
            bits[idx] = bits[idx] + step          # +-1 ulp on the bf16 bit pattern
            out = bits.view(torch.bfloat16).reshape(tb.shape).contiguous()
    cur[k] = out
    nb = out.numel() * out.element_size()
    cur_bytes += nb
    if cur_bytes >= limit:
        flush()
flush()
json.dump({"metadata": {"total_size": sum(os.path.getsize(os.path.join(a.out, s)) for s in shard_names)},
           "weight_map": weight_map},
          open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
print(f"[ulp-control] wrote {a.out}: {len(keys)} tensors, "
      f"perturbed {tot_changed}/{tot_elems} = {tot_changed/max(1,tot_elems):.2%} of weights by +-1 bf16 ulp")
