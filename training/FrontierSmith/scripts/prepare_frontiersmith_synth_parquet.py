#!/usr/bin/env python3
"""
Prepare the colleague's `frontiersmith_synth` problem set (500 open-ended
optimization problems) as a VERL RL-training Parquet, MIRRORING the FrontierCS
wiring (scripts/prepare_frontiercs_parquet.py).

Each row:
  prompt        = [{"role":"user","content": <task derived from statement>}]
  reward_model  = {"ground_truth": <problem id, e.g. "fsx_S_0001">}
  data_source   = "frontiersmith_synth"
  extra_info    = {"scoring_shape": "gen_checker"|"evaluator",
                   "lang": "cpp"|"py", "format": <meta.format>, "prefix": <fsx_X>}

Scoring shapes (see frontiersmith_synth/harness/*):
  * gen_checker  (config.checker is a .cc/.cpp OR a .py that is NOT evaluator.py,
                  with a gen.cpp/gen.py present): the candidate is a stdin->stdout
                  PROGRAM. For each of n_cases the harness generates an input,
                  runs the candidate, then runs `checker in out ans` which prints
                  `Ratio: <float in [0,1]>`. Score = mean ratio. (formats A/C/D)
                  -> candidate language is C++ when solutions are .cpp (testlib
                     formats A/"?"), Python when solutions are .py (formats C/D).
  * evaluator    (config.checker == evaluator.py, no gen.*): the candidate IS a
                  Python program that reads ONE public-instance JSON on stdin and
                  writes ONE JSON answer on stdout; the evaluator runs it via
                  isorun over a fixed instance distribution and prints
                  `Ratio: <mean in [0,1]>`. (formats B/E)  candidate language = py.

The reward (verl/verl/utils/reward_score/frontiersmith_synth.py) reuses the
harness to score, so a row is "runnable" iff the harness can score it -- which is
exactly the set whose validation.json verdict == PASS. All 500 currently PASS.

Usage:
  python scripts/prepare_frontiersmith_synth_parquet.py                 # all runnable
  python scripts/prepare_frontiersmith_synth_parquet.py --cpp-only      # testlib C++ only
  python scripts/prepare_frontiersmith_synth_parquet.py --val-ratio 0.1 # train/val split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# The synth tree lives OUTSIDE FrontierSmith; resolve it relative to this repo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_ROOT_DEFAULT = (
    PROJECT_ROOT.parent / "innovation_prior" / "frontiersmith_synth"
)
DEFAULT_OUT = PROJECT_ROOT / "data" / "frontiersmith_synth"


def _read_config_checker(pdir: Path) -> str:
    """Return the `checker:` filename from config.yaml (the harness resolves this)."""
    import re

    txt = (pdir / "config.yaml").read_text()
    m = re.search(r"checker:\s*(\S+)", txt)
    return m.group(1) if m else ""


def classify(pdir: Path) -> dict | None:
    """Determine scoring shape + candidate language for a problem dir.

    Returns a dict {scoring_shape, lang, checker} or None if the dir is not a
    runnable synth problem (missing files). Mirrors the harness's own resolution:
      - evaluator.py checker + no gen.* -> program-mode (validate_pyproblem.py)
      - gen.cpp/gen.py + a checker (.cc/.cpp/.py) -> gen+checker (validate_problem.py)
    """
    if not (pdir / "config.yaml").exists():
        return None
    checker = _read_config_checker(pdir)
    if not checker:
        return None
    has_gen = (pdir / "gen.cpp").exists() or (pdir / "gen.py").exists()

    # program-mode: evaluator.py, no separate generator.
    if checker == "evaluator.py" and (pdir / "evaluator.py").exists() and not has_gen:
        return {"scoring_shape": "evaluator", "lang": "py", "checker": checker}

    # gen + checker: need a generator and a resolvable checker file.
    if has_gen and (pdir / checker).exists():
        # candidate language = language of the reference solutions.
        sol_dir = pdir / "solutions"
        sols = list(sol_dir.glob("*.cpp")) + list(sol_dir.glob("*.py")) if sol_dir.exists() else []
        if not sols:
            return None
        n_cpp = sum(1 for s in sols if s.suffix == ".cpp")
        n_py = sum(1 for s in sols if s.suffix == ".py")
        lang = "cpp" if n_cpp >= n_py else "py"
        return {"scoring_shape": "gen_checker", "lang": lang, "checker": checker}

    return None


def _read_statement(pdir: Path) -> str | None:
    for name in ("statement.md", "statement.txt"):
        p = pdir / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    return None


def build_prompt(statement: str, shape: str, lang: str) -> list[dict]:
    """Build the user message. The instruction matches the candidate contract the
    harness enforces so a well-formed rollout is actually gradeable."""
    if shape == "evaluator":
        # Program-mode: candidate reads ONE JSON instance on stdin, writes ONE JSON
        # answer on stdout (routed through isorun by the evaluator).
        instr = (
            "You are solving an open-ended optimization problem. Write a COMPLETE "
            "Python 3 program that reads ONE JSON \"public instance\" from stdin and "
            "prints ONE JSON answer to stdout (the last JSON value printed is read). "
            "Any feasible output is scored; better constructions score strictly higher. "
            "No network, no file access. Output ONLY the Python code wrapped in "
            "```python and ```. No explanation."
        )
        fence = "python"
    elif lang == "cpp":
        instr = (
            "You are solving an open-ended optimization problem. Write a COMPLETE "
            "C++17 program that reads the test case from standard input and writes a "
            "feasible solution to standard output in the format the statement requires. "
            "Any feasible output is scored; better constructions score strictly higher. "
            "Output ONLY the C++ code wrapped in ```cpp and ```. No explanation."
        )
        fence = "cpp"
    else:  # gen_checker, python candidate
        instr = (
            "You are solving an open-ended optimization problem. Write a COMPLETE "
            "Python 3 program that reads the test case from standard input and writes a "
            "feasible solution to standard output in the format the statement requires. "
            "Any feasible output is scored; better constructions score strictly higher. "
            "Output ONLY the Python code wrapped in ```python and ```. No explanation."
        )
        fence = "python"
    return [
        {
            "role": "user",
            "content": f"{instr}\n\n{statement}\n\nGenerate the solution (```{fence} ... ```):",
        }
    ]


def build_rows(synth_root: Path, cpp_only: bool, py_only: bool,
               evaluator_only: bool, gen_checker_only: bool) -> list[dict]:
    problems_dir = synth_root / "problems"
    rows: list[dict] = []
    skipped: dict[str, int] = {}
    for pdir in sorted(problems_dir.iterdir()):
        if not pdir.is_dir():
            continue
        pid = pdir.name
        info = classify(pdir)
        if info is None:
            skipped[pid] = skipped.get(pid, 0) + 1
            continue
        if cpp_only and info["lang"] != "cpp":
            continue
        if py_only and info["lang"] != "py":
            continue
        if evaluator_only and info["scoring_shape"] != "evaluator":
            continue
        if gen_checker_only and info["scoring_shape"] != "gen_checker":
            continue
        statement = _read_statement(pdir)
        if statement is None:
            skipped[pid] = skipped.get(pid, 0) + 1
            continue
        meta = {}
        try:
            meta = json.loads((pdir / "meta.json").read_text())
        except Exception:
            pass
        rows.append(
            {
                "prompt": build_prompt(statement, info["scoring_shape"], info["lang"]),
                "reward_model": {"ground_truth": pid},
                "data_source": "frontiersmith_synth",
                "extra_info": {
                    "scoring_shape": info["scoring_shape"],
                    "lang": info["lang"],
                    "checker": info["checker"],
                    "format": meta.get("format", "?"),
                    "prefix": pid.rsplit("_", 1)[0],
                },
            }
        )
    if skipped:
        print(f"[warn] skipped {len(skipped)} unrunnable problem dirs: "
              f"{sorted(skipped)[:10]}{' ...' if len(skipped) > 10 else ''}",
              file=sys.stderr)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-root", type=Path, default=SYNTH_ROOT_DEFAULT,
                    help="path to the frontiersmith_synth tree")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--val-ratio", type=float, default=0.0,
                    help="fraction held out as val.parquet (0 => only full.parquet)")
    ap.add_argument("--cpp-only", action="store_true", help="only C++/testlib candidates")
    ap.add_argument("--py-only", action="store_true", help="only Python candidates")
    ap.add_argument("--evaluator-only", action="store_true", help="only program-mode (evaluator.py)")
    ap.add_argument("--gen-checker-only", action="store_true", help="only gen+checker problems")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    synth_root = args.synth_root.resolve()
    if not (synth_root / "problems").is_dir():
        sys.exit(f"ERROR: {synth_root}/problems not found")

    rows = build_rows(synth_root, args.cpp_only, args.py_only,
                      args.evaluator_only, args.gen_checker_only)
    if not rows:
        print("No runnable problems found.")
        return

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Always write full.parquet (used for both train & val, like frontiercs full.parquet).
    full_path = args.output_dir / "full.parquet"
    df.to_parquet(full_path, index=False)
    print(f"Saved {len(df)} problems -> {full_path}")

    # train.parquet == full (or a train/val split when --val-ratio > 0).
    if args.val_ratio > 0:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        n_val = max(1, int(len(df) * args.val_ratio))
        df.iloc[n_val:].to_parquet(args.output_dir / "train.parquet", index=False)
        df.iloc[:n_val].to_parquet(args.output_dir / "val.parquet", index=False)
        print(f"Saved {len(df) - n_val} train + {n_val} val -> {args.output_dir}")
    else:
        df.to_parquet(args.output_dir / "train.parquet", index=False)
        print(f"Saved {len(df)} train (== full) -> {args.output_dir / 'train.parquet'}")

    # Breakdown by (scoring_shape, lang, prefix).
    from collections import Counter
    by_shape = Counter((r["extra_info"]["scoring_shape"], r["extra_info"]["lang"]) for r in rows)
    by_prefix = Counter(r["extra_info"]["prefix"] for r in rows)
    print("  by (scoring_shape, lang):", dict(by_shape))
    print("  by prefix:", dict(sorted(by_prefix.items())))


if __name__ == "__main__":
    main()
