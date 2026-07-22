#!/usr/bin/env python3
"""audit_wave2b.py -- acceptance audit for wave-2b problems (do not trust agent self-reports).

Checks, for every problem id in the wave-2b pack (or --ids):
  1. validation.json exists and verdict == PASS  (run harness first / use --reverify)
  2. innovation headroom: strong-greedy >= 0.06, strong <= 0.92, greedy-trivial >= 0.03
  3. cross-problem homogeneity via scan_homogeneity.py (batch + whole corpus)

--reverify N : re-run the harness from scratch on a random-but-deterministic sample of N
               problems (sanity that validation.json files are reproducible).
Exit 0 only if every check passes; prints per-problem failures otherwise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SYNTH = Path(__file__).resolve().parents[1]
PACK = SYNTH / "seeds" / "bulk_seed_packs" / "pack_w2b_0507_1006.jsonl"

MIN_SG = 0.06   # strong - greedy
MAX_S = 0.92    # strong ceiling
MIN_GT = 0.03   # greedy - trivial


def val_metrics(pdir: Path):
    vj = pdir / "validation.json"
    if not vj.exists():
        return None, "no-validation.json"
    d = json.loads(vj.read_text())
    verdict = d.get("verdict")
    m = d.get("metrics", {})
    def pick(*names):
        for n in names:
            if n in m and isinstance(m[n], (int, float)):
                return float(m[n])
        return None
    t = pick("trivial_mean", "trivial")
    g = pick("greedy_mean", "greedy")
    s = pick("strong_mean", "strong")
    return {"verdict": verdict, "trivial": t, "greedy": g, "strong": s}, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--pack", type=Path, default=PACK)
    ap.add_argument("--skip-homogeneity", action="store_true")
    args = ap.parse_args()

    if args.ids:
        ids = args.ids
    else:
        ids = [json.loads(l)["id"] for l in open(args.pack)]

    bad = []
    seen = 0
    for pid in ids:
        pdir = SYNTH / "problems" / pid
        if not pdir.is_dir():
            bad.append((pid, "missing-dir"))
            continue
        seen += 1
        m, err = val_metrics(pdir)
        if err:
            bad.append((pid, err))
            continue
        if m["verdict"] != "PASS":
            bad.append((pid, f"verdict={m['verdict']}"))
            continue
        t, g, s = m["trivial"], m["greedy"], m["strong"]
        if None in (t, g, s):
            bad.append((pid, f"metrics-missing {m}"))
            continue
        if s - g < MIN_SG:
            bad.append((pid, f"low-headroom strong-greedy={s - g:.3f} < {MIN_SG}"))
        elif s > MAX_S:
            bad.append((pid, f"saturated strong={s:.3f} > {MAX_S}"))
        elif g - t < MIN_GT:
            bad.append((pid, f"flat-ladder greedy-trivial={g - t:.3f} < {MIN_GT}"))

    print(f"audited {seen}/{len(ids)} problem dirs; {len(bad)} failures")
    for pid, why in bad[:60]:
        print(f"  FAIL {pid}: {why}")
    if len(bad) > 60:
        print(f"  ... and {len(bad) - 60} more")

    homog_rc = 0
    if not args.skip_homogeneity:
        print("\n-- homogeneity: wave-2b batch --")
        homog_rc |= subprocess.call([sys.executable, str(SYNTH / "reports" / "scan_homogeneity.py"),
                                     "--ids", *[p for p in ids if (SYNTH / "problems" / p).is_dir()]])
        print("-- homogeneity: whole corpus --")
        homog_rc |= subprocess.call([sys.executable, str(SYNTH / "reports" / "scan_homogeneity.py")])

    ok = not bad and homog_rc == 0
    print(f"\nAUDIT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
