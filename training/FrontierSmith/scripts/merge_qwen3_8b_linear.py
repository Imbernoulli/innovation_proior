#!/usr/bin/env python3
"""Linearly merge local Qwen3-8B and Qwen3-8B-Base checkpoints.

For each alpha, this writes:

    merged = alpha * Qwen3-8B + (1 - alpha) * Qwen3-8B-Base

Only floating-point tensors are mixed. Non-floating tensors, if any, are copied
from Qwen3-8B after verifying the Base tensor matches exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INSTRUCT = PROJECT_ROOT / "models" / "Qwen3-8B"
DEFAULT_BASE = PROJECT_ROOT / "models" / "Qwen3-8B-Base"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models"


def parse_alpha(value: str) -> float:
    alpha = float(value)
    if not 0.0 <= alpha <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be between 0 and 1")
    return alpha


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def read_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shard_metadata(path: Path) -> dict[str, str] | None:
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.metadata()


def copy_sidecars(instruct_dir: Path, output_dir: Path) -> None:
    for src in instruct_dir.iterdir():
        if src.name.endswith(".safetensors"):
            continue
        dst = output_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def validate_indexes(instruct_index: dict, base_index: dict) -> list[str]:
    instruct_map = instruct_index.get("weight_map", {})
    base_map = base_index.get("weight_map", {})
    if instruct_map != base_map:
        missing = sorted(set(instruct_map) ^ set(base_map))
        raise ValueError(f"weight maps differ; symmetric diff has {len(missing)} keys")
    return sorted(set(instruct_map.values()))


def merge_tensor(name: str, instruct_tensor: torch.Tensor, base_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    if instruct_tensor.shape != base_tensor.shape:
        raise ValueError(f"{name}: shape mismatch {tuple(instruct_tensor.shape)} vs {tuple(base_tensor.shape)}")
    if instruct_tensor.dtype != base_tensor.dtype:
        raise ValueError(f"{name}: dtype mismatch {instruct_tensor.dtype} vs {base_tensor.dtype}")

    if torch.is_floating_point(instruct_tensor):
        merged = instruct_tensor.float().mul(alpha).add_(base_tensor.float(), alpha=1.0 - alpha)
        return merged.to(dtype=instruct_tensor.dtype)

    if not torch.equal(instruct_tensor, base_tensor):
        raise ValueError(f"{name}: non-floating tensor differs between source models")
    return instruct_tensor


def merge_one_alpha(instruct_dir: Path, base_dir: Path, output_dir: Path, alpha: float, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    instruct_index = read_index(instruct_dir)
    base_index = read_index(base_dir)
    shard_names = validate_indexes(instruct_index, base_index)

    copy_sidecars(instruct_dir, output_dir)
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(
            {
                "alpha": alpha,
                "formula": "merged = alpha * Qwen3-8B + (1 - alpha) * Qwen3-8B-Base",
                "instruct_model": str(instruct_dir),
                "base_model": str(base_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for shard_name in shard_names:
        instruct_path = instruct_dir / shard_name
        base_path = base_dir / shard_name
        output_path = output_dir / shard_name

        merged_tensors = {}
        with safe_open(instruct_path, framework="pt", device="cpu") as instruct_file:
            with safe_open(base_path, framework="pt", device="cpu") as base_file:
                instruct_keys = set(instruct_file.keys())
                base_keys = set(base_file.keys())
                if instruct_keys != base_keys:
                    raise ValueError(f"{shard_name}: tensor keys differ")
                for name in sorted(instruct_keys):
                    merged_tensors[name] = merge_tensor(
                        name,
                        instruct_file.get_tensor(name),
                        base_file.get_tensor(name),
                        alpha,
                    )
        save_file(merged_tensors, output_path, metadata=shard_metadata(instruct_path))
        print(f"[alpha={alpha:.2f}] wrote {output_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruct-dir", type=Path, default=DEFAULT_INSTRUCT)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alphas", type=parse_alpha, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for model_dir in (args.instruct_dir, args.base_dir):
        if not (model_dir / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"missing model index in {model_dir}")

    for alpha in args.alphas:
        out_dir = args.output_root / f"Qwen3-8B-linear-alpha-{alpha_tag(alpha)}"
        merge_one_alpha(args.instruct_dir, args.base_dir, out_dir, alpha, args.overwrite)


if __name__ == "__main__":
    main()
