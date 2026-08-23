#!/usr/bin/env python3
"""Derive PROTOCOL-FAITHFUL ThetaEvolve OpenEvolve configs from the official it_XL
configs, changing ONLY what a local, single-model, 1-GPU eval genuinely requires.

WHY THIS EXISTS
---------------
The audit of our ThetaEvolve eval (cc_eval_theta_openevolve_ailab.sh) found the
scoring is faithful (same evaluator_modular.py + verify.py; combined_score is the
official per-task metric), but the checked-in `config_<task>_qwen35_local_smoke.yaml`
files drifted from the official `config_<task>_it_XL.yaml` protocol in TWO ways that
change *what the model is asked to do*:

  1. num_top_programs / num_diverse_programs: official 0/0 -> smoke 1/1 (all tasks).
     The official AlphaEvolve-style loop gives the model NO in-context exemplars
     (empty {evolution_history} block); the smoke injects a "Prior programs" +
     "Diverse programs" block into every prompt. (openevolve/prompt/sampler.py
     gates these blocks on num_top_programs/num_diverse_programs.)
  2. system_message_list content was rewritten/thinned on several tasks
     (circle_packing & hadamard dropped strategy catalogs; first_autocorr ADDED a
     message; second_autocorr condensed messages).

This generator emits `config_<task>_qwen35_faithful.yaml` that keeps the OFFICIAL:
  - prompt block  (system_message_list VERBATIM, num_top/diverse/inspiration = 0/0/0,
                   all prompt flags, max_artifact_bytes)
  - variables / score_transform / core_parameters (the scoring inputs)
  - database structure (feature_dimensions, elite/exploitation ratios,
                        diff_based_evolution, allow_full_rewrites)
  - evaluator structure (cascade, use_llm_feedback, artifacts)

...and OVERRIDES only the local-environment necessities (each override is inherent
to running a local single model on a 1-GPU, 55-min Slurm job, NOT protocol drift):
  - llm.models  -> a single local vLLM model (name from --served-model-name),
                   weight 1.0, api_base -> the local vLLM (patched again at runtime
                   by the launcher's sed so VLLM_PORT stays authoritative).
                   (The official 0.8/0.2 gemini ensemble cannot be a local model.)
  - llm.max_tokens -> keep the OFFICIAL 16384 (do NOT drop to 12288; thinking-mode
                      Qwen3.5 truncates diffs below ~16k -> "No valid diffs found").
  - max_iterations / population_size / archive_size / num_islands /
    parallel_evaluations -> sized so a real evolutionary run FITS the job budget
                            (defaults chosen to be a meaningful search, tunable via
                            env / CLI). These are SCALE, not protocol: they change
                            how MUCH search happens, not what the model is asked.
  - MAX_RUNTIME / evaluator.timeout -> capped so a single program eval cannot eat
                            the whole Slurm wallclock. (Bounds reachable optimum,
                            not the score definition.)

The launcher (cc_eval_theta_openevolve_ailab.sh) still sed-patches api_base at run
time, so VLLM_PORT remains authoritative; nothing about serving changes.

Existing `*_qwen35_local_smoke.yaml` configs and every prior run are LEFT UNTOUCHED.

Usage:
  python make_theta_faithful_configs.py                 # write all 5 faithful configs
  python make_theta_faithful_configs.py --task hadamard_matrix
  SERVED_MODEL_NAME=qwen35-9b python make_theta_faithful_configs.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

try:
    import yaml
except Exception as e:  # pragma: no cover
    print(f"ERROR: PyYAML required ({e})", file=sys.stderr)
    raise SystemExit(2)

THETA_ROOT = Path("/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve")
EX_ROOT = THETA_ROOT / "openevolve_adapted" / "examples"

TASKS = [
    "circle_packing_modular",
    "first_autocorr_inequality",
    "second_autocorr_inequality",
    "third_autocorr_inequality",
    "hadamard_matrix",
]

# Local-environment overrides. These are SCALE / serving knobs, NOT protocol.
# Defaults are a genuine (small-but-real) search that fits a 1-GPU 55-min job.
DEFAULTS = dict(
    served_model_name=os.environ.get("SERVED_MODEL_NAME", "qwen35-9b"),
    # api_base is re-patched by the launcher's sed at runtime; this is just a
    # sane placeholder so a standalone run still points at a local server.
    api_base=os.environ.get("FAITHFUL_API_BASE", "http://127.0.0.1:8021/v1"),
    max_iterations=int(os.environ.get("FAITHFUL_MAX_ITERATIONS", "40")),
    population_size=int(os.environ.get("FAITHFUL_POPULATION_SIZE", "1000")),
    archive_size=int(os.environ.get("FAITHFUL_ARCHIVE_SIZE", "100")),
    num_islands=int(os.environ.get("FAITHFUL_NUM_ISLANDS", "4")),
    parallel_evaluations=int(os.environ.get("FAITHFUL_PARALLEL_EVALS", "2")),
    checkpoint_interval=int(os.environ.get("FAITHFUL_CHECKPOINT_INTERVAL", "10")),
    # Cap a single program's runtime so one eval can't eat the whole Slurm job.
    max_program_runtime=int(os.environ.get("FAITHFUL_MAX_RUNTIME", "60")),
    evaluator_timeout=int(os.environ.get("FAITHFUL_EVALUATOR_TIMEOUT", "90")),
    # Keep the OFFICIAL token budget; thinking-mode Qwen truncates below ~16k.
    max_tokens=int(os.environ.get("FAITHFUL_MAX_TOKENS", "16384")),
    random_seed=int(os.environ.get("SEED", "3407")),
)


def build_faithful(cfg: dict, d: dict) -> dict:
    """Transform a loaded it_XL config dict into a faithful-local config dict.

    Preserves the official prompt (system_message_list, num_top/diverse/inspiration,
    flags), variables/score_transform, database structure and evaluator structure.
    Overrides ONLY the local-env necessities described in the module docstring.
    """
    out = copy.deepcopy(cfg)

    # --- top-level: add a bounded iteration budget + seed (official leaves these
    #     to slime / the RL driver; a standalone OpenEvolve run needs them). ---
    out["max_iterations"] = d["max_iterations"]
    out["random_seed"] = d["random_seed"]
    out["checkpoint_interval"] = d["checkpoint_interval"]

    # --- variables.MAX_RUNTIME: cap per-program runtime (bounds reachable optimum,
    #     not the score formula). Only lower it, never raise above official. ---
    vlist = out.setdefault("variables", {})
    if "MAX_RUNTIME" in vlist:
        try:
            vlist["MAX_RUNTIME"] = min(int(vlist["MAX_RUNTIME"]), d["max_program_runtime"])
        except Exception:
            vlist["MAX_RUNTIME"] = d["max_program_runtime"]

    # --- llm: single local model (the ONLY unavoidable protocol substitution). ---
    llm = out.setdefault("llm", {})
    llm["models"] = [{"name": d["served_model_name"], "weight": 1.0}]
    # Use the SAME local model as the evaluator model (official uses gemini for both;
    # use_llm_feedback is false in every task config, so evaluator_models is unused
    # for scoring, but keep it consistent so no accidental remote call is attempted).
    llm["evaluator_models"] = [{"name": d["served_model_name"], "weight": 1.0}]
    llm["api_base"] = d["api_base"]
    llm["api_key"] = "EMPTY"
    # Preserve official temperature / top_p (they match already); keep official 16384.
    llm["max_tokens"] = d["max_tokens"]
    llm.setdefault("temperature", 0.7)
    llm.setdefault("top_p", 0.95)

    # --- prompt: LEAVE VERBATIM. Force the official exemplar counts in case an
    #     upstream base ever changed them (official it_XL = 0/0/0). ---
    prompt = out.setdefault("prompt", {})
    prompt["num_top_programs"] = 0
    prompt["num_diverse_programs"] = 0
    prompt["num_inspiration_programs"] = 0

    # --- database: keep official STRUCTURE, scale down SIZE to fit the job. ---
    db = out.setdefault("database", {})
    db["population_size"] = d["population_size"]
    db["archive_size"] = d["archive_size"]
    db["num_islands"] = d["num_islands"]
    # feature_dimensions / elite_selection_ratio / exploitation_ratio kept as-is.

    # --- evaluator: keep official structure, scale parallelism + cap timeout. ---
    ev = out.setdefault("evaluator", {})
    ev["parallel_evaluations"] = d["parallel_evaluations"]
    ev["timeout"] = d["evaluator_timeout"]

    return out


HEADER = (
    "# AUTO-GENERATED by FrontierSmith/scripts/make_theta_faithful_configs.py\n"
    "# PROTOCOL-FAITHFUL ThetaEvolve config: derived from config_{task}_it_XL.yaml.\n"
    "# Prompt (system_message_list, num_top/diverse/inspiration=0/0/0, flags) and\n"
    "# scoring (variables/score_transform + evaluator_modular.py) are the OFFICIAL\n"
    "# ones. Only the local-env necessities are overridden: single local vLLM model,\n"
    "# bounded max_iterations/population/parallelism, and a per-program runtime cap.\n"
    "# See the generator script docstring for the full rationale. DO NOT hand-edit;\n"
    "# regenerate instead. The launcher sed-patches api_base so VLLM_PORT is authoritative.\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", help="task name or 'all'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    d = DEFAULTS
    print(f"[faithful] served_model_name={d['served_model_name']} "
          f"max_iterations={d['max_iterations']} population={d['population_size']} "
          f"islands={d['num_islands']} max_tokens={d['max_tokens']}", file=sys.stderr)

    written = []
    for task in tasks:
        base = EX_ROOT / task / "configs" / f"config_{task}_it_XL.yaml"
        if not base.is_file():
            print(f"ERROR: missing official base {base}", file=sys.stderr)
            return 2
        cfg = yaml.safe_load(base.read_text())
        faithful = build_faithful(cfg, d)
        text = HEADER.replace("{task}", task) + yaml.safe_dump(
            faithful, sort_keys=False, allow_unicode=True, width=100000
        )
        out_path = EX_ROOT / task / "configs" / f"config_{task}_qwen35_faithful.yaml"
        if args.dry_run:
            print(f"--- would write {out_path} ---")
            print(text)
        else:
            out_path.write_text(text)
            print(f"wrote {out_path}")
        written.append(str(out_path))

    if not args.dry_run:
        print(f"[faithful] {len(written)} config(s) written.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
