#!/usr/bin/env python3
"""Fix HF safetensors key names produced by merge_fsdp_to_hf.py for Qwen3.5-9B.

Qwen3.5-9B is a multimodal `Qwen3_5ForConditionalGeneration` whose LM submodule is
nested as `language_model.model.*` in the verl FSDP checkpoint. merge_fsdp_to_hf.py
(validated on the dense Qwen3-8B) emits keys like
    model.language_model.model.layers.0.input_layernorm.weight
but the canonical HF format (and what vLLM expects) is
    model.language_model.layers.0.input_layernorm.weight
i.e. one extra `.model.` segment after `language_model`. vLLM then reports those
params as "not initialized from checkpoint" and the engine fails to start.

This script rewrites the merged safetensors with the corrected key names into a new
dir, pulling canonical config/tokenizer from the original base model. No re-merge,
no GPU; only key strings change (tensor data/shapes/dtypes untouched).

Usage:
  python scripts/cc_fix_qwen35_hf_keys.py \
      --src models/cc_qwen35_9b_mixed_hf \
      --dst models/cc_qwen35_9b_mixed_hf_fixed \
      --base models/Qwen3.5-9B
"""
import argparse, glob, json, os, shutil
from safetensors.torch import load_file, save_file


def fix_key(k: str) -> str:
    return k.replace("language_model.model.", "language_model.", 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="merged HF dir from merge_fsdp_to_hf.py")
    ap.add_argument("--dst", required=True, help="output dir for corrected model")
    ap.add_argument("--base", required=True, help="original base model dir (canonical config/tokenizer)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    # canonical config/tokenizer/etc. from base (skip its weights + index)
    for f in sorted(glob.glob(os.path.join(args.base, "*"))):
        b = os.path.basename(f)
        if b.endswith(".safetensors") or b.endswith(".safetensors.index.json"):
            continue
        if os.path.isfile(f):
            shutil.copy2(f, os.path.join(args.dst, b))

    shards = sorted(glob.glob(os.path.join(args.src, "model-*.safetensors"))) or \
        sorted(glob.glob(os.path.join(args.src, "model.safetensors")))
    assert shards, f"no safetensors shards in {args.src}"

    weight_map, total_size, nrenamed = {}, 0, 0
    for sp in shards:
        name = os.path.basename(sp)
        nt = {}
        for k, v in load_file(sp).items():
            nk = fix_key(k)
            nrenamed += int(nk != k)
            nt[nk] = v
            weight_map[nk] = name
            total_size += v.numel() * v.element_size()
        save_file(nt, os.path.join(args.dst, name), metadata={"format": "pt"})
        print(f"  {name}: {len(nt)} params")

    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map},
              open(os.path.join(args.dst, "model.safetensors.index.json"), "w"), indent=2)
    print(f"renamed {nrenamed} keys; {len(weight_map)} keys; {total_size/1e9:.2f} GB -> {args.dst}")


if __name__ == "__main__":
    main()
