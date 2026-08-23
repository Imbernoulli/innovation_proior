#!/usr/bin/env python3
# Copyright 2025 - Precompute per-problem ALE-Bench raw-score references.
"""Precompute the per-problem reference scale used to normalize ALE-Bench rewards.

Mixed GRPO mixes FrontierCS (0-100) with ALE-Bench full whose raw
``overall_absolute_score`` is unbounded (1e4..1e13) and inverted for
minimize-type problems. ``verl/verl/utils/reward_score/reward_norm.py`` maps the
raw score to a bounded 0-100 reward via a direction-aware saturating ratio
against a per-problem reference ``R`` (roughly the human-best raw score on the
public case scale). This script computes ``R`` once, offline, from the ALE-Bench
problem data zips and writes the table consumed by the reward at train time.

Reference derivation (raw ABSOLUTE scale, matching public_eval which always
returns the raw "Score = N" sum regardless of relative_score_type):
  * absolute-score problems (no relative_results): use standings_scores.csv,
    top human cumulative private score / num_private_cases * num_public_cases.
  * relative-score problems: use relative_results.csv (raw absolute private
    scores per case per participant), take the per-case best (max for maximize,
    min positive for minimize), median over cases, * num_public_cases.

Run on a login node (data zips are pre-staged under $ALE_BENCH_DATA):
  python scripts/prepare_alebench_score_references.py \
    --data-dir data/alebench/local_data \
    --out data/alebench/ale_score_references.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import zipfile
from pathlib import Path


def _member(z: zipfile.ZipFile, suffix: str) -> str | None:
    for n in z.namelist():
        if n.endswith(suffix) and not n.endswith("/"):
            return n
    return None


def _read_csv_rows(z: zipfile.ZipFile, name: str) -> list[dict]:
    with z.open(name) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))


def compute_reference(z: zipfile.ZipFile) -> dict:
    data = json.loads(z.read(_member(z, "data.json")))
    md = data["metadata"]
    seeds = data["seeds"]
    npub = len(seeds["public"])
    npriv = len(seeds["private"])
    score_type = md["score_type"]  # "maximize" | "minimize"

    rel_name = _member(z, "relative_results.csv")
    if rel_name is not None:
        rows = _read_csv_rows(z, rel_name)
        cols = list(rows[0].keys()) if rows else []  # one column per private case
        per_case_best = []
        for c in cols:
            vals = [int(r[c]) for r in rows if str(r.get(c, "")).strip() not in ("", "-1")]
            vals = [v for v in vals if v >= 0]
            if not vals:
                continue
            if score_type == "maximize":
                per_case_best.append(max(vals))
            else:
                pos = [v for v in vals if v > 0]
                per_case_best.append(min(pos) if pos else 0)
        per_case = statistics.median(per_case_best) if per_case_best else 0.0
        reference = float(per_case) * npub
        source = "relative_results(raw_abs per-case)"
    else:
        rows = _read_csv_rows(z, _member(z, "standings_scores.csv"))
        scores = sorted([int(r["score"]) for r in rows if str(r.get("score", "")).strip()], reverse=True)
        best = scores[0] if scores else 0
        reference = float(best) / max(npriv, 1) * npub
        source = "standings(raw_abs cumulative)"

    return {
        "score_type": score_type,
        "num_public_cases": npub,
        "num_private_cases": npriv,
        "reference": reference,
        "source": source,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("ALE_BENCH_DATA", "data/alebench/local_data"))
    ap.add_argument("--out", default="data/alebench/ale_score_references.json")
    ap.add_argument("--problems", nargs="*", default=None, help="optional subset of problem ids")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.problems:
        pids = args.problems
    else:
        pids = sorted(p.stem for p in data_dir.glob("*.zip"))

    table: dict[str, dict] = {}
    for pid in pids:
        zip_path = data_dir / f"{pid}.zip"
        if not zip_path.is_file():
            print(f"[skip] {pid}: {zip_path} not found")
            continue
        with zipfile.ZipFile(zip_path) as z:
            table[pid] = compute_reference(z)
        t = table[pid]
        print(f"{pid:<26}{t['score_type']:<10}ref={t['reference']:>20,.0f}  {t['source']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(table, fh, indent=2)
    print(f"\nWrote {len(table)} references -> {out}")


if __name__ == "__main__":
    main()
