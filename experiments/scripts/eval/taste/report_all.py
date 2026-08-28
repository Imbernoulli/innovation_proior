#!/usr/bin/env python3
"""One command -> the whole taste-suite table.

    python report_all.py --dir outputs_taste/run1 --judged-dir outputs_taste/judged \
        --arms base wd01 soup_a10 [--exclude .cache/taste_eval/contaminated_ids.json]

Generation benchmarks (giants / abgen / hypoarena) are read from the JUDGED files;
everything else from the raw generation files.  SciJudge splits are intersected
across arms so a capped or resumed run never inflates one column.
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import SCORERS, load, paired_boot, _drop  # noqa: E402

# (label, scorer key, file template, intersect?)
BENCHES = [
    ("GiantsBench (insight, judge 1-10)",        "giants",        "{jd}/judged_taste_{arm}.jsonl",        False),
    ("HypoArena (hypothesis, judge 1-10)",       "hypoarena",     "{jd}/judged_hypoarena_{arm}.jsonl",    False),
    ("AbGen (ablation design, judge 1-10)",      "abgen",         "{jd}/judged_abgen_{arm}.jsonl",        False),
    ("SciJudge main / thinking",                 "scijudge",      "{d}/cc_eval_taste_{arm}_scijudge.jsonl",         True),
    ("SciJudge main / no-thinking",              "scijudge",      "{d}/cc_eval_taste_{arm}_scijudge_nothink.jsonl", True),
    ("SciJudge OOD-year 2025",                   "scijudge",      "{d}/cc_eval_taste_{arm}_scijudge_oodyear.jsonl", True),
    ("SciJudge OOD-ICLR review score",           "scijudge_iclr", "{d}/cc_eval_taste_{arm}_scijudge_iclr.jsonl",    True),
    ("RINoBench (novelty 1-5)",                  "rino",          "{d}/cc_eval_taste_{arm}_rino.jsonl",   False),
    ("SoundnessBench (rigor bucket)",            "soundness",     "{d}/cc_eval_taste_{arm}_soundness.jsonl", False),
]

HEAD = {
    "giants":    ("mean(1-10)", lambda s: f"{s['mean_similarity']:.2f}", lambda s: f"no-insight {s['n_no_insight_floored_to_1']}/{s['n_items']}"),
    "abgen":     ("mean(1-10)", lambda s: f"{s['mean_similarity']:.2f}", lambda s: f"no-answer {s['n_no_insight_floored_to_1']}/{s['n_items']}"),
    "hypoarena": ("mean(1-10)", lambda s: f"{s['mean_similarity']:.2f}", lambda s: f"no-answer {s['n_no_insight_floored_to_1']}/{s['n_items']}"),
    "scijudge":  ("swap-consistent acc", lambda s: f"{s['consistency_acc']:.1%}", lambda s: f"no-ans {s['no_answer_rate']:.1%} / trunc {s['truncated_rate']:.1%}"),
    "scijudge_iclr": ("swap-consistent acc", lambda s: f"{s['consistency_acc']:.1%}", lambda s: f"no-ans {s['no_answer_rate']:.1%} / trunc {s['truncated_rate']:.1%}"),
    "rino":      ("Spearman rho", lambda s: f"{s['spearman_rho']:.3f}", lambda s: f"exact {s['exact_acc_over_parsed']:.1%} / MAE {s['mae']:.2f}"),
    "soundness": ("macro-F1", lambda s: f"{s['macro_f1']:.3f}", lambda s: f"FPR {s['false_positive_rate_low_called_high']:.1%} / low-R {s['low_recall']:.1%}"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--judged-dir", required=True)
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--exclude", default=None)
ap.add_argument("--json-out", default=None)
a = ap.parse_args()

out = {}
for label, key, tpl, do_int in BENCHES:
    files = {arm: tpl.format(d=a.dir, jd=a.judged_dir, arm=arm) for arm in a.arms}
    have = {arm: f for arm, f in files.items() if os.path.exists(f) and os.path.getsize(f) > 0}
    if not have:
        continue
    loaded = {arm: _drop(load(f), key, a.exclude) for arm, f in have.items()}
    note = ""
    if do_int and len(loaded) > 1:
        def keys(rows):
            seen = defaultdict(set)
            for r in rows:
                seen[r["meta"]["pair"]].add(bool(r["meta"]["swap"]))
            return {k for k, v in seen.items() if len(v) == 2}
        common = set.intersection(*(keys(rows) for rows, _ in loaded.values()))
        note = f"n={len(common)} pairs (intersected)"
        loaded = {arm: ([r for r in rows if r["meta"]["pair"] in common], d)
                  for arm, (rows, d) in loaded.items()}
    summ, items = {}, {}
    for arm, (rows, _) in loaded.items():
        summ[arm], items[arm] = SCORERS[key](rows)
    out[label] = {"key": key, "note": note, "summary": summ, "items": items}

hdr, main, side = "metric", None, None
print()
print("| benchmark | metric | " + " | ".join(a.arms) + " | detail (" + a.arms[0] + " / " + " / ".join(a.arms[1:]) + ") |")
print("|---|---|" + "---:|" * len(a.arms) + "---|")
for label, blob in out.items():
    name, fmt, det = HEAD[blob["key"]]
    cells, dets = [], []
    for arm in a.arms:
        s = blob["summary"].get(arm)
        cells.append(fmt(s) if s else "-")
        dets.append(det(s) if s else "-")
    n = blob["note"]
    print(f"| {label}{(' — ' + n) if n else ''} | {name} | " + " | ".join(cells) + " | " + " ; ".join(dets) + " |")

print("\n**paired bootstrap vs " + a.arms[0] + "**\n")
for label, blob in out.items():
    base = a.arms[0]
    if base not in blob["items"]:
        continue
    for arm in a.arms[1:]:
        if arm not in blob["items"]:
            continue
        r = paired_boot(blob["items"][base], blob["items"][arm])
        if r:
            star = " **sig**" if r["p_two_sided"] < 0.05 else ""
            print(f"- {label} — {arm}: Δ={r['delta']:+.4f} CI95=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] p={r['p_two_sided']:.3f} (n={r['n']}){star}")
if a.json_out:
    json.dump({k: v["summary"] for k, v in out.items()}, open(a.json_out, "w"), indent=2, default=str)
    print(f"\n[report] json -> {a.json_out}")
