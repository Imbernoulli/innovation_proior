#!/usr/bin/env python3
"""Metrics for the taste-eval suite.

  python score.py scijudge  gen_a.jsonl [gen_b.jsonl ...]
  python score.py soundness gen_a.jsonl [...]
  python score.py giants    judged_a.jsonl [...]

Scoring follows each benchmark's own protocol:
  scijudge  -- position-swap consistency: a pair counts as correct only when the
               model picks the higher-cited paper in BOTH orders (SciJudgeBench
               §评分协议).  Chance level is 25%, not 50%.
  soundness -- binary rigor bucket, accuracy / macro-F1 / Cohen kappa exactly as
               in rigorbench.evaluation.metrics, plus the paper's headline
               false-positive rate (low-rigor items called "high").
  giants    -- mean 1-10 similarity from the LM judge.

Unparseable outputs are NEVER silently dropped: they are counted as wrong (the
benchmarks score them that way) and reported separately as `no_answer`.
When two or more files are given, the last-vs-first pair also gets a paired
bootstrap over the shared item ids.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from benches import extract_ab, extract_bucket, extract_insight, extract_rating_1_5, extract_letter  # noqa: E402


def load(path):
    """Last record wins per id -- the generation files are append-only, so a
    resumed or accidentally duplicated run leaves several lines for one id."""
    seen = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        seen[r["id"]] = r
    return list(seen.values())


def boot_ci(vals, n=5000, seed=7):
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(vals)
    xs = []
    for _ in range(n):
        xs.append(sum(vals[rng.randrange(m)] for _ in range(m)) / m)
    xs.sort()
    return xs[int(0.025 * n)], xs[int(0.975 * n)]


def paired_boot(a: dict, b: dict, n=5000, seed=7):
    """P(mean(b) > mean(a)) style two-sided p over shared ids."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    m = len(d)
    obs = sum(d) / m
    cnt = 0
    for _ in range(n):
        s = sum(d[rng.randrange(m)] for _ in range(m)) / m
        if (s <= 0) if obs > 0 else (s >= 0):
            cnt += 1
    lo, hi = boot_ci(d, n, seed)
    return {"n": m, "delta": obs, "ci95": [lo, hi], "p_two_sided": min(1.0, 2 * cnt / n)}


# ------------------------------- scijudge ---------------------------------- #
def score_scijudge(rows):
    by_pair = defaultdict(dict)
    for r in rows:
        pid = r["meta"]["pair"]
        by_pair[pid]["swap" if r["meta"]["swap"] else "plain"] = r
    per_item, cats = {}, defaultdict(list)
    n_pairs = n_complete = 0
    no_ans = 0
    plain_ok = swap_ok = 0
    picked_A = 0
    tot_side = 0
    trunc = 0
    for pid, d in sorted(by_pair.items()):
        n_pairs += 1
        if "plain" not in d or "swap" not in d:
            continue
        n_complete += 1
        ok_both = True
        for k in ("plain", "swap"):
            r = d[k]
            a = extract_ab(r.get("output") or "")
            tot_side += 1
            if r.get("finish_reason") == "length":
                trunc += 1
            if a is None:
                no_ans += 1
                ok_both = False
                continue
            if a == "A":
                picked_A += 1
            good = a == r["meta"]["gold"]
            if k == "plain":
                plain_ok += good
            else:
                swap_ok += good
            ok_both &= good
        per_item[pid] = 1.0 if ok_both else 0.0
        cats[d["plain"]["meta"].get("category") or "?"].append(per_item[pid])
    vals = list(per_item.values())
    acc = sum(vals) / len(vals) if vals else float("nan")
    lo, hi = boot_ci(vals)
    return {
        "metric": "position-swap consistency accuracy (chance 25%)",
        "n_pairs": n_complete,
        "consistency_acc": acc,
        "ci95": [lo, hi],
        "plain_order_acc": plain_ok / max(1, n_complete),
        "swap_order_acc": swap_ok / max(1, n_complete),
        "picked_A_rate": picked_A / max(1, tot_side - no_ans),
        "no_answer_rate": no_ans / max(1, tot_side),
        "truncated_rate": trunc / max(1, tot_side),
        "by_category": {k: round(sum(v) / len(v), 4) for k, v in sorted(cats.items())},
    }, per_item


# ------------------------------- soundness --------------------------------- #
def _kappa(pred, gold):
    n = len(pred)
    if n < 2:
        return None
    labels = sorted(set(pred) | set(gold))
    obs = sum(p == g for p, g in zip(pred, gold)) / n
    exp = sum((pred.count(l) / n) * (gold.count(l) / n) for l in labels)
    return 1.0 if exp == 1.0 else (obs - exp) / (1 - exp)


def score_soundness(rows):
    per_item = {}
    pred_l, gold_l = [], []
    unparsed = trunc = 0
    confs = []
    for r in rows:
        gold = r["meta"]["gold"]
        b, c = extract_bucket(r.get("output") or "")
        if r.get("finish_reason") == "length":
            trunc += 1
        if b is None:
            unparsed += 1
            per_item[r["id"]] = 0.0
            continue
        if c is not None:
            confs.append(c)
        pred_l.append(b)
        gold_l.append(gold)
        per_item[r["id"]] = 1.0 if b == gold else 0.0
    n = len(pred_l)
    def prf(lbl):
        tp = sum(p == lbl and g == lbl for p, g in zip(pred_l, gold_l))
        fp = sum(p == lbl and g != lbl for p, g in zip(pred_l, gold_l))
        fn = sum(p != lbl and g == lbl for p, g in zip(pred_l, gold_l))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        return pr, rc, f1
    ph, rh, fh = prf("high")
    pl, rl, fl = prf("low")
    n_low = sum(g == "low" for g in gold_l)
    fpr = sum(p == "high" and g == "low" for p, g in zip(pred_l, gold_l)) / max(1, n_low)
    vals = list(per_item.values())
    lo, hi = boot_ci(vals)
    return {
        "metric": "binary rigor bucket (direct_bucket prompt)",
        "n_scored": len(rows),
        "n_parsed": n,
        "accuracy_over_all": sum(vals) / max(1, len(vals)),
        "ci95": [lo, hi],
        "accuracy_over_parsed": sum(p == g for p, g in zip(pred_l, gold_l)) / max(1, n),
        "macro_f1": (fh + fl) / 2,
        "high_precision": ph, "high_recall": rh, "high_f1": fh,
        "low_precision": pl, "low_recall": rl, "low_f1": fl,
        "false_positive_rate_low_called_high": fpr,
        "cohen_kappa": _kappa(pred_l, gold_l),
        "pred_dist": dict(Counter(pred_l)),
        "gold_dist": dict(Counter(gold_l)),
        "unparseable_rate": unparsed / max(1, len(rows)),
        "truncated_rate": trunc / max(1, len(rows)),
        "mean_confidence": sum(confs) / len(confs) if confs else None,
    }, per_item


# -------------------------------- giants ----------------------------------- #
def score_giants(rows):
    """Mean 1-10 similarity.

    An item the model never produced an insight for (ran past max_tokens still
    mid-essay, or emitted nothing extractable) is scored at the rubric FLOOR of
    1.0 rather than dropped -- dropping it would reward whichever arm truncates
    most.  Judge-infrastructure errors are excluded instead, since those are our
    fault, not the model's.  Both readings are reported.
    """
    per_item, doms = {}, defaultdict(list)
    judged, floored, infra_err = {}, 0, 0
    for r in rows:
        v = r.get("rating")
        dom = (r.get("meta") or {}).get("domain") or "?"
        if v is None:
            if r.get("reason") == "no_insight":
                floored += 1
                per_item[r["id"]] = 1.0
                doms[dom].append(1.0)
            else:
                infra_err += 1
            continue
        judged[r["id"]] = float(v)
        per_item[r["id"]] = float(v)
        doms[dom].append(float(v))
    jv = list(judged.values())
    av = list(per_item.values())
    lo, hi = boot_ci(av)
    return {
        "metric": "LM-judge similarity to ground-truth insight (1-10)",
        "n_items": len(av),
        "n_judged": len(jv),
        "n_no_insight_floored_to_1": floored,
        "n_judge_infra_errors_excluded": infra_err,
        "mean_similarity": sum(av) / max(1, len(av)),
        "ci95": [lo, hi],
        "mean_over_judged_only": sum(jv) / max(1, len(jv)),
        "by_domain": {k: round(sum(v) / len(v), 3) for k, v in sorted(doms.items())},
        "hist": {str(k): v for k, v in sorted(Counter(round(x) for x in av).items())},
    }, per_item


SCORERS = {"scijudge": score_scijudge, "soundness": score_soundness, "giants": score_giants}

# --------------------------------- rino ------------------------------------ #
def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(x) < 2:
        return None
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def score_rino(rows):
    """Novelty rating 1-5 against RINoBench's human labels.

    Exact-match accuracy is a harsh read of an ordinal label, so within-1
    accuracy, MAE and Spearman rho are reported alongside it.  An unparseable
    answer counts as wrong for accuracy and is excluded from rho/MAE (and its
    count is printed).
    """
    per_item = {}
    preds, golds = [], []
    unparsed = trunc = 0
    for r in rows:
        gold = int(r["meta"]["gold"])
        if r.get("finish_reason") == "length":
            trunc += 1
        p = extract_rating_1_5(r.get("output") or "")
        if p is None:
            unparsed += 1
            per_item[r["id"]] = 0.0
            continue
        preds.append(p)
        golds.append(gold)
        per_item[r["id"]] = 1.0 if p == gold else 0.0
    n = len(preds)
    within1 = sum(abs(p - g) <= 1 for p, g in zip(preds, golds)) / max(1, n)
    mae = sum(abs(p - g) for p, g in zip(preds, golds)) / max(1, n)
    vals = list(per_item.values())
    lo, hi = boot_ci(vals)
    return {
        "metric": "novelty rating 1-5 vs RINoBench human label",
        "n_scored": len(rows),
        "n_parsed": n,
        "exact_acc_over_all": sum(vals) / max(1, len(vals)),
        "ci95": [lo, hi],
        "exact_acc_over_parsed": sum(p == g for p, g in zip(preds, golds)) / max(1, n),
        "within_1_acc": within1,
        "mae": mae,
        "spearman_rho": _spearman(preds, golds),
        "pred_dist": {str(k): v for k, v in sorted(Counter(preds).items())},
        "gold_dist": {str(k): v for k, v in sorted(Counter(golds).items())},
        "mean_pred": sum(preds) / max(1, n),
        "mean_gold": sum(golds) / max(1, n),
        "unparseable_rate": unparsed / max(1, len(rows)),
        "truncated_rate": trunc / max(1, len(rows)),
    }, per_item


def score_scipredict(rows):
    """Multiple-choice experiment-outcome prediction (SciPredict MCQ subset).

    Objective GT, no judge.  Chance is 1/(number of options); the released items
    are mostly 4-way, so ~25%.  Unparseable answers count as wrong and are
    reported separately.
    """
    per_item, doms = {}, defaultdict(list)
    unparsed = trunc = 0
    for r in rows:
        gold = str(r["meta"]["gold"]).upper()
        if r.get("finish_reason") == "length":
            trunc += 1
        p = extract_letter(r.get("output") or "")
        ok = 0.0
        if p is None:
            unparsed += 1
        else:
            ok = 1.0 if p == gold else 0.0
        per_item[r["id"]] = ok
        doms[r["meta"].get("domain") or "?"].append(ok)
    vals = list(per_item.values())
    lo, hi = boot_ci(vals)
    return {
        "metric": "MCQ accuracy on experiment-outcome prediction (chance ~25%)",
        "n": len(vals),
        "accuracy": sum(vals) / max(1, len(vals)),
        "ci95": [lo, hi],
        "by_domain": {k: round(sum(v) / len(v), 4) for k, v in sorted(doms.items())},
        "unparseable_rate": unparsed / max(1, len(rows)),
        "truncated_rate": trunc / max(1, len(rows)),
    }, per_item


SCORERS["rino"] = score_rino
SCORERS["pairiq"] = score_scijudge
SCORERS["scipredict"] = score_scipredict
SCORERS["scijudge_iclr"] = score_scijudge
SCORERS["abgen"] = score_giants
SCORERS["hypoarena"] = score_giants


def _drop(rows, bench, excl_path):
    """Drop benchmark items whose paper is in the innovation training corpus."""
    if not excl_path:
        return rows, 0
    bad = set(json.load(open(excl_path)).get(bench, []))
    if not bad:
        return rows, 0
    if bench == "scijudge":
        keep = [r for r in rows if r["meta"]["pair"] not in bad]
    else:
        keep = [r for r in rows if r["id"] not in bad]
    return keep, len(rows) - len(keep)


if __name__ == "__main__":
    bench = sys.argv[1]
    argv = sys.argv[2:]
    excl = None
    if "--exclude" in argv:
        i = argv.index("--exclude")
        excl = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    intersect = False
    if "--intersect" in argv:
        argv.remove("--intersect")
        intersect = True
    paths = argv
    loaded = {p: _drop(load(p), bench, excl) for p in paths}
    if intersect and len(paths) > 1:
        # Arms stopped at different points (a run capped at a common prefix, a
        # resumed arm that got further).  Restrict every arm to the items ALL of
        # them finished, so the table compares like with like.
        def keys(rows):
            if not (bench.startswith("scijudge") or bench == "pairiq"):
                return {r["id"] for r in rows}
            # a pair only counts as finished when BOTH orders are present
            seen = defaultdict(set)
            for r in rows:
                seen[r["meta"]["pair"]].add(bool(r["meta"]["swap"]))
            return {k for k, v in seen.items() if len(v) == 2}
        common = set.intersection(*(keys(rows) for rows, _ in loaded.values()))
        print(f"[intersect] {len(common)} items completed by all {len(paths)} arms")
        for p in paths:
            rows, d = loaded[p]
            k = "pair" if (bench.startswith("scijudge") or bench == "pairiq") else None
            loaded[p] = ([r for r in rows if (r["meta"]["pair"] if k else r["id"]) in common], d)
    per = {}
    for p in paths:
        rows, dropped = loaded[p]
        if dropped:
            print(f"[decontam] {p}: dropped {dropped} contaminated records")
        summ, item = SCORERS[bench](rows)
        per[p] = item
        print("=" * 78)
        print(p)
        print(json.dumps(summ, indent=2, ensure_ascii=False))
    if len(paths) >= 2:
        print("=" * 78)
        for p in paths[1:]:
            r = paired_boot(per[paths[0]], per[p])
            print(f"paired bootstrap  {p}  vs  {paths[0]}: {json.dumps(r)}")
