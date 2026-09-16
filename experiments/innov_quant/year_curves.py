#!/usr/bin/env python3
"""Per-year score and completion curves, so the year hypothesis can be looked at directly.

year_shape.py answers the hypothesis as a test (NEAR-FAR contrast, quadratic sign) and
year_completion.py does the same for completion, but neither of them ever prints the
curve itself.  A reader asked to believe "there is no peak at 2025" should be able to see
the line.  This emits the raw per-year means, plus the JSON a chart can be drawn from.

TWO THINGS THAT WOULD MAKE THE CURVE LIE IF THEY WERE NOT HANDLED

  Different problem sets per year.  A year whose directory is missing a few problems would
  move the mean for a reason that has nothing to do with the year.  Every point of a curve
  is therefore restricted to the problems present in ALL of that (arm, bench)'s year
  directories, and the surviving count is printed next to the curve.

  The contaminated pair.  `rlv5_lo32nm_a10_s20`'s first y1950 and y2025 runs were assembled
  after an 8h walltime TIMEOUT and are survivor-biased toward short generations (section
  28).  Their clean r2 re-runs are substituted, same as year_shape.py does -- y2025 is the
  NEAR point, so using the contaminated one would bend the curve exactly where the
  hypothesis is being tested.

Scores weight problems equally: mean within a problem over its draws, then mean over
problems.  Completion is dump2.py's definition, `</think>` present in the text.
"""
import collections
import glob
import json
import os

import numpy as np

OUT = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = {("rlv5_lo32nm_a10_s20", 1950): "rlv5_lo32nm_a10_s20_y1950r2",
         ("rlv5_lo32nm_a10_s20", 2025): "rlv5_lo32nm_a10_s20_y2025r2"}
GRID = {
    "base9b_v2c":             [1700, 1800, 1900, 1950, 1975, 2000, 2010, 2025, 2026, 2050, 2075, 2100],
    "lo32nm_a10":             [1700, 1800, 1900, 1950, 1975, 2000, 2010, 2025, 2026, 2050, 2075, 2100],
    "rlv5_lo32nm_a10_s20":    [1800, 1900, 1950, 1975, 2000, 2010, 2025, 2026, 2050, 2075, 2100],
    "base4b":                 [1900, 2000, 2010, 2026, 2100],
    "4b_lo32nm_a10":          [1900, 2000, 2010, 2026, 2100],
    "rlv5_4b_lo32nm_a10_s20": [1900, 2000, 2010, 2026, 2100],
    "ft01mix_a10":            [2000, 2025, 2050, 2075],
    "rlv5_base_s20":          [2000, 2025, 2050, 2075],
    "rlv5_ft01mix_a10_s20":   [2000, 2025, 2050, 2075],
    "4b_ft01mix_a10":         [2000, 2025, 2050, 2075],
    "rlv5_4b_base_s20":       [2000, 2025, 2050, 2075],
    "rlv5_4b_ft01mix_a10_s20": [2000, 2025, 2050, 2075],
}
SWEEP = {a: ("new" if y == [2000, 2025, 2050, 2075] else "old") for a, y in GRID.items()}
FAM = {a: ("4B" if a.startswith(("base4b", "4b_", "rlv5_4b")) else "9B") for a in GRID}
BEN = ["frontiercs", "alebench"]


def read(tag):
    """-> {bench: {problem: {"s": [scores], "c": [0/1]}}}"""
    got = collections.defaultdict(lambda: collections.defaultdict(lambda: {"s": [], "c": []}))
    pat = os.path.join(OUT, f"cc_eval_{tag}_thinking_32k_both_vllm", "shard_*", "samples.jsonl")
    for f in glob.glob(pat):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("error") is not None:
                continue
            gt = r.get("ground_truth")
            gt = gt if isinstance(gt, str) else json.dumps(gt, sort_keys=True)
            cell = got[r.get("data_source")][gt]
            s = (r.get("metrics") or {}).get("score")
            if s is not None:
                cell["s"].append(float(s))
            cell["c"].append(1.0 if "</think>" in (r.get("text") or "") else 0.0)
    return got


def main():
    curves = {}
    for arm, years in GRID.items():
        per_year = {}
        for y in years:
            tag = CLEAN.get((arm, y)) or f"{arm}_y{y}"
            per_year[y] = read(tag)
        for b in BEN:
            sets = [set(per_year[y].get(b, {})) for y in years]
            if not sets or not all(sets):
                continue
            common = set.intersection(*sets)
            if len(common) < 8:
                continue
            sc, cp = [], []
            for y in years:
                d = per_year[y][b]
                sc.append(float(np.mean([np.mean(d[p]["s"]) for p in common if d[p]["s"]])))
                cp.append(float(np.mean([np.mean(d[p]["c"]) for p in common if d[p]["c"]])))
            curves[f"{arm}|{b}"] = dict(arm=arm, bench=b, fam=FAM[arm], sweep=SWEEP[arm],
                                        years=years, n_prob=len(common),
                                        score=[round(v, 4) for v in sc],
                                        comp=[round(v, 4) for v in cp])
            print(f"{arm:26s} {b:11s} n={len(common):4d}  "
                  f"score {' '.join(f'{v:7.3f}' for v in sc)}")
    json.dump(curves, open(os.path.join(HERE, "year_curves.json"), "w"), indent=1)
    lines = ["# 逐年曲线原始值", "",
             "每条曲线只用该 (臂, bench) **所有年份目录都有**的题,逐题等权。",
             "`rlv5_lo32nm_a10_s20` 的 y1950 / y2025 用干净的 r2 复跑(§28)。", ""]
    for key in sorted(curves):
        c = curves[key]
        lines += [f"### `{c['arm']}` · {c['bench']} · n={c['n_prob']} 题", "",
                  "| 年份 | " + " | ".join(str(y) for y in c["years"]) + " |",
                  "|---|" + "---|" * len(c["years"]),
                  "| 分数 | " + " | ".join(f"{v:.3f}" for v in c["score"]) + " |",
                  "| 完成率 | " + " | ".join(f"{v:.3f}" for v in c["comp"]) + " |", ""]
    open(os.path.join(HERE, "year_curves.md"), "w").write("\n".join(lines) + "\n")
    print(f"\nwrote year_curves.json ({len(curves)} curves) + year_curves.md")


if __name__ == "__main__":
    main()
