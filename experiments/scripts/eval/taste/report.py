#!/usr/bin/env python3
"""Assemble the three taste benchmarks into one markdown table.

    python report.py --dir outputs_taste/run1 --judged-dir outputs_taste/judged \
        --arms taste_base taste_wd01 taste_soup_a10 [--exclude contaminated_ids.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import SCORERS, load, paired_boot, _drop  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--judged-dir", default=None)
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--exclude", default=None)
ap.add_argument("--json-out", default=None)
a = ap.parse_args()

BENCH_FILE = {
    "soundness": lambda arm: os.path.join(a.dir, f"cc_eval_{arm}_soundness.jsonl"),
    "scijudge": lambda arm: os.path.join(a.dir, f"cc_eval_{arm}_scijudge.jsonl"),
    "giants": lambda arm: os.path.join(a.judged_dir or a.dir, f"judged_{arm}.jsonl"),
}

out = {}
for bench in ("giants", "scijudge", "soundness"):
    out[bench] = {}
    items = {}
    for arm in a.arms:
        f = BENCH_FILE[bench](arm)
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            continue
        rows, dropped = _drop(load(f), bench, a.exclude)
        summ, item = SCORERS[bench](rows)
        summ["_dropped_contaminated"] = dropped
        out[bench][arm] = summ
        items[arm] = item
    base = a.arms[0]
    for arm in a.arms[1:]:
        if arm in items and base in items:
            out[bench].setdefault("_paired_vs_" + base, {})[arm] = paired_boot(items[base], items[arm])

def f(x, n=3):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))

lines = []
g = out["giants"]
if any(k in g for k in a.arms):
    lines += ["", "### GiantsBench — insight anticipation (LM judge 1-10)", "",
              "| arm | n | mean (no-insight=1) | 95% CI | mean over judged | no-insight | `<insight>` tag rate |",
              "|---|---:|---:|---|---:|---:|---:|"]
    for arm in a.arms:
        s = g.get(arm)
        if not s:
            continue
        gen = os.path.join(a.judged_dir or a.dir, f"gen_{arm}.jsonl")
        tag = "-"
        if os.path.exists(gen):
            rr = load(gen)
            tag = f(sum("<insight>" in (x.get("output") or "").lower() for x in rr) / max(1, len(rr)))
        lines.append(f"| {arm} | {s['n_items']} | **{f(s['mean_similarity'])}** | "
                     f"[{f(s['ci95'][0])}, {f(s['ci95'][1])}] | {f(s['mean_over_judged_only'])} | "
                     f"{s['n_no_insight_floored_to_1']} | {tag} |")
s = out["scijudge"]
if any(k in s for k in a.arms):
    lines += ["", "### SciJudgeBench — citation-impact judgment (swap-consistent acc, chance 25%)", "",
              "| arm | pairs | consistency acc | 95% CI | plain | swap | picked-A | no-answer | trunc |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for arm in a.arms:
        r = s.get(arm)
        if not r:
            continue
        lines.append(f"| {arm} | {r['n_pairs']} | **{f(r['consistency_acc'])}** | "
                     f"[{f(r['ci95'][0])}, {f(r['ci95'][1])}] | {f(r['plain_order_acc'])} | "
                     f"{f(r['swap_order_acc'])} | {f(r['picked_A_rate'])} | "
                     f"{f(r['no_answer_rate'])} | {f(r['truncated_rate'])} |")
b = out["soundness"]
if any(k in b for k in a.arms):
    lines += ["", "### SoundnessBench — proposal soundness judgment", "",
              "| arm | n | macro-F1 | acc(parsed) | high-R | low-R | FPR | kappa | unparseable |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in a.arms:
        r = b.get(arm)
        if not r:
            continue
        lines.append(f"| {arm} | {r['n_parsed']} | **{f(r['macro_f1'])}** | {f(r['accuracy_over_parsed'])} | "
                     f"{f(r['high_recall'])} | {f(r['low_recall'])} | "
                     f"{f(r['false_positive_rate_low_called_high'])} | {f(r['cohen_kappa'])} | "
                     f"{f(r['unparseable_rate'])} |")
lines += ["", "### Paired bootstrap vs " + a.arms[0], ""]
for bench in ("giants", "scijudge", "soundness"):
    for arm, d in (out[bench].get("_paired_vs_" + a.arms[0], {}) or {}).items():
        if d:
            lines.append(f"- **{bench}** {arm}: Δ={d['delta']:+.4f} "
                         f"CI95=[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] p={d['p_two_sided']:.3f} (n={d['n']})")
print("\n".join(lines))
if a.json_out:
    json.dump(out, open(a.json_out, "w"), indent=2)
    print(f"\n[report] json -> {a.json_out}")
