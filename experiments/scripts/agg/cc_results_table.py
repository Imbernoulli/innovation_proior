#!/usr/bin/env python3
"""Aggregate the matrix campaign evals into ONE comparison table.

Scans the completed eval summaries and joins them by model tag:
  FCS   = metrics.frontiercs.score.mean@5   (FrontierCS, higher better)
  ALE   = metrics.alebench.score.mean@5     (ALE-Bench performance, higher better)
  Theta = best_combined_score               (circle-packing, higher better ~2.0-2.6)
  TTT   = best_combined_score               (AC3 third-autocorr, higher better)

Tags follow the orchestrator scheme:
  q35_a{00,20,50,80,100}                         start model (premerge ratio = % instruct)
  q35_a{..}_{innovonly,innovmaint}               SFT model
  q35_a{..}_{innovonly,innovmaint}_soupa{50,70,90}  postmerge soup (sft fraction)
Run anytime; it just reports whatever has completed so far.
"""
import json, glob, os, re

ROOT = "/scratch/gpfs/CHIJ/bohan/fs"
FS = f"{ROOT}/FrontierSmith"
THETA = f"{ROOT}/ThetaEvolve"

def _mean5(metrics, group, key="score"):
    try:
        return metrics[group][key]["mean@5"]
    except Exception:
        return None

def load_fcsale():
    out = {}
    for p in glob.glob(f"{FS}/outputs/cc_eval_*_thinking_32k_both_vllm/summary.json"):
        tag = re.search(r"cc_eval_(.+?)_thinking_32k_both_vllm", p).group(1)
        try:
            m = json.load(open(p)).get("metrics", {})
        except Exception:
            continue
        # ALE "score"(=performance) has a ~310 failure-FLOOR; the real signal is the
        # absolute score (0 => submissions don't compile/score). Carry both.
        ale_abs = _mean5(m, "alebench", "overall_absolute_score")
        out[tag] = {"FCS": _mean5(m, "frontiercs"), "ALE": _mean5(m, "alebench"),
                    "ALEabs": ale_abs}
    return out

def load_theta(prefix, label):
    """prefix='cc_eval_theta_' (circle) or 'cc_eval_theta_ttt_' (ttt)."""
    out = {}
    for p in glob.glob(f"{THETA}/outputs/{prefix}*/**/summary.json", recursive=True):
        base = os.path.basename(os.path.dirname(os.path.dirname(p)))  # cc_eval_theta_<tag>_<task>
        try:
            d = json.load(open(p))
        except Exception:
            continue
        tag = d.get("tag")
        if not tag:
            continue
        out[tag] = d.get("best_combined_score")
    return out

fcsale = load_fcsale()
# circle: prefix cc_eval_theta_ but NOT cc_eval_theta_ttt_
theta_all = load_theta("cc_eval_theta_", "theta")
ttt_all = load_theta("cc_eval_theta_ttt_", "ttt")
# theta_all also captured ttt entries (prefix overlap) -> strip those whose tag is in ttt naming
# (ttt summaries carry tag like 'ttt_<tag>'? verify by separating on path); fall back to dir prefix:
def load_theta_strict(circle=True):
    out = {}
    for p in glob.glob(f"{THETA}/outputs/cc_eval_theta_*/**/summary.json", recursive=True):
        topdir = p.split("/outputs/")[1].split("/")[0]  # cc_eval_theta[_ttt]_<tag>_<task>
        is_ttt = topdir.startswith("cc_eval_theta_ttt_")
        if circle and is_ttt: continue
        if (not circle) and (not is_ttt): continue
        try: d = json.load(open(p))
        except Exception: continue
        tag = d.get("tag")
        if tag and tag.startswith("ttt_"):   # ttt summaries prefix the tag with 'ttt_'
            tag = tag[4:]
        if tag: out[tag] = d.get("best_combined_score")
    return out

theta = load_theta_strict(circle=True)
ttt = load_theta_strict(circle=False)

tags = sorted(set(fcsale) | set(theta) | set(ttt))
# order: start models, then sft, then soups; q35_a00.. ascending
def sortkey(t):
    m = re.match(r"q35_a(\d+)(?:_(innov\w+?))?(?:_soupa(\d+))?$", t)
    if not m: return (9, t)
    a = int(m.group(1)); data = m.group(2) or ""; soup = int(m.group(3)) if m.group(3) else -1
    kind = 0 if not data else (1 if soup < 0 else 2)
    return (kind, a, data, soup)

def f(x, w=8, p=3):
    return (f"%{w}.{p}f" % x) if isinstance(x, (int, float)) else (" " * (w - 1) + "-")

def ale_cell(r):
    """ALE score, flagged '!' when absolute score is 0 => submissions broken (floor ~310)."""
    s = r.get("ALE"); a = r.get("ALEabs")
    if not isinstance(s, (int, float)):
        return "       -"
    broken = isinstance(a, (int, float)) and a <= 0.0
    return f"{s:7.1f}{'!' if broken else ' '}"

print(f"\n{'model tag':<34} {'FCS':>8} {'ALE':>8} {'Theta':>8} {'TTT':>8}    (ALE '!' = broken, abs=0)")
print("-" * 80)
prev = None
for t in sorted(tags, key=sortkey):
    if not t.startswith("q35"):
        continue
    fam = (re.match(r"(q35_a\d+)", t) or [None, t])[1] if re.match(r"(q35_a\d+)", t) else t
    fam = re.match(r"(q35_a\d+)", t).group(1) if re.match(r"(q35_a\d+)", t) else t
    if prev is not None and fam != prev:
        print()
    prev = fam
    r = {**fcsale.get(t, {}), "Theta": theta.get(t), "TTT": ttt.get(t)}
    print(f"{t:<34} {f(r.get('FCS'))} {ale_cell(r)} {f(r.get('Theta'))} {f(r.get('TTT'))}")
print("-" * 80)
n = lambda d: len([t for t in d if t.startswith('q35')])
print(f"models evaluated: FCS+ALE={n(fcsale)}  Theta={n(theta)}  TTT={n(ttt)}  (matrix complete = 45 models)")
