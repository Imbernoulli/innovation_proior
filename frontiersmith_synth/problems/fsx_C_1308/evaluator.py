#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1308 -- "Reviewing What Matters With Too Few Reviewers"
(family: content-moderation-policy; format B, quality-metric).

THEME.  A moderation queue has far more items than the review team can look at in a
shift.  Every item already carries a moderation MODEL's violation-probability score
and a hidden ground-truth label.  Items that are never sent to a human are
auto-actioned by a fixed rule (remove iff model_score >= auto_threshold); that rule
is right most of the time but not always.  Sending an item to a reviewer instead
buys a probabilistic re-decision whose accuracy depends on THAT reviewer's base
skill, how many items they have already reviewed in a row this shift (fatigue,
without a rotation break), and how genuinely ambiguous the item is.  Getting a
decision wrong costs something -- missing a real violation costs its category's
SEVERITY, wrongly removing a fine item costs its category's APPEAL cost.  The
policy maker (the candidate) must pick which of the too-many items to spend the
too-few reviewer-minutes on, and in what order per reviewer, to minimize total
expected harm+appeal cost.

MECHANISMS COMPOSED.
  - harm-severity-weighting : missing a true violation costs categories["..."]["severity"].
  - reviewer-fatigue-accuracy: eff_acc = base_accuracy - fatigue_rate * (queue position)
    for the reviewer that item is placed with, floored at floor_acc.
  - appeal-cost             : wrongly removing a non-violation costs categories["..."]["appeal_cost"].
  decision-uncertainty (ambiguity, public per item) ALSO erodes eff_acc, and jointly
  with severity/appeal magnitude determines whether sending an item to review even
  helps: reviewing an item the auto-rule was already very likely to get right can
  make the EXPECTED cost worse (probabilistic review risks reversing a correct call).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the schema.
  stdout: ONE JSON object {"schedule": {"<reviewer_id>": [item_id, ...], ...}}
          where the list for a reviewer is the ORDER that reviewer works the items
          (position 0 = first / freshest). Every item id appears at most once
          across the whole schedule; a reviewer's list length must not exceed
          that reviewer's capacity. Any item id absent from every list is
          auto-actioned. Malformed output -> that instance scores 0.0.

SCORING (deterministic; no wall-time). For every item the evaluator computes the
REALIZED cost given a schedule:
    - not reviewed: cost = 0 if (model_score>=auto_threshold) == true_violation
                    else severity/appeal_cost of its category
    - reviewed at (reviewer r, position k): cost = (1 - eff_acc(r,k,item)) * loss
                    where loss = severity if truly a violation else appeal_cost
Summed over all items -> total_cost(schedule). The evaluator ALSO computes, itself:
    cost_auto   = total_cost of the empty schedule (nothing reviewed)         -- weak anchor
    cost_oracle = total_cost of an internal reference policy that (unlike any
                  candidate) is allowed to look at the HIDDEN true labels to
                  greedily pick a strong (not necessarily optimal) subset +
                  assignment                                                  -- strong anchor
  r = clamp( 0.1 + 0.9 * (cost_auto - cost_cand) / max(1e-6, cost_auto - cost_oracle), 0, 1 )
A schedule matching "do nothing" scores ~0.1; approaching the hidden-info oracle
scores near 1.0 (loose, not fully reachable without the hidden labels -> headroom).
Doing WORSE than doing nothing (e.g. burning reviewers on items the model was
already confidently correct about) scores BELOW 0.1.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance. Hidden true labels
and the oracle reference are computed in THIS parent process only.

CLI: python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

MASK = (1 << 64) - 1


def _rng(seed):
    state = seed & MASK

    def _step():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK
        return state

    def nxt_float():
        return (_step() >> 11) / float(1 << 53)

    def nxt_int(lo, hi):
        return lo + (_step() >> 17) % (hi - lo + 1)

    return nxt_float, nxt_int


# ----------------------------- category table -------------------------------
# severity/appeal_cost are the tunable constants (live in the PUBLIC instance,
# not hardcoded in the statement); base_rate/noise/ambig shape the GENERATOR only.
CAT_NAMES = ["spam", "misinfo", "harassment", "hate_speech", "graphic_violence"]
CATS = {
    "spam":             {"severity": 0.10, "appeal_cost": 0.06, "base_rate": 0.85, "noise": 0.06, "ambig_base": 0.10, "ambig_noise": 0.08},
    "misinfo":          {"severity": 0.28, "appeal_cost": 0.17, "base_rate": 0.55, "noise": 0.18, "ambig_base": 0.35, "ambig_noise": 0.15},
    "harassment":       {"severity": 0.40, "appeal_cost": 0.24, "base_rate": 0.50, "noise": 0.22, "ambig_base": 0.45, "ambig_noise": 0.15},
    "hate_speech":      {"severity": 0.55, "appeal_cost": 0.30, "base_rate": 0.45, "noise": 0.30, "ambig_base": 0.55, "ambig_noise": 0.15},
    "graphic_violence": {"severity": 0.60, "appeal_cost": 0.32, "base_rate": 0.40, "noise": 0.25, "ambig_base": 0.50, "ambig_noise": 0.15},
}

AUTO_THRESHOLD = 0.5
FLOOR_ACC = 0.74
DIFFICULTY_PENALTY = 0.12

# ----------------------------- instance family -------------------------------
# (seed, N, [(base_accuracy, fatigue_rate, capacity), ...])
SPECS = [
    (101, 50, [(0.95, 0.020, 6), (0.90, 0.025, 6), (0.86, 0.030, 5)]),
    (102, 55, [(0.94, 0.018, 7), (0.89, 0.025, 6), (0.85, 0.030, 5)]),
    (201, 60, [(0.96, 0.015, 5), (0.92, 0.022, 5), (0.87, 0.028, 5), (0.83, 0.035, 5)]),
    (202, 45, [(0.95, 0.022, 5), (0.88, 0.028, 5)]),
    (203, 65, [(0.96, 0.018, 8), (0.90, 0.022, 7), (0.85, 0.030, 6)]),
    # -- larger / harder held-out instances below --
    (301, 70, [(0.97, 0.014, 6), (0.93, 0.018, 6), (0.88, 0.022, 6), (0.84, 0.030, 6), (0.80, 0.038, 5)]),
    (302, 40, [(0.98, 0.010, 4), (0.93, 0.022, 4), (0.88, 0.028, 4)]),
    (401, 75, [(0.94, 0.022, 9), (0.89, 0.028, 8), (0.84, 0.032, 7)]),
    (402, 48, [(0.97, 0.015, 5), (0.87, 0.035, 5), (0.81, 0.040, 4)]),
    (501, 80, [(0.95, 0.018, 10), (0.91, 0.022, 9), (0.86, 0.028, 8), (0.81, 0.035, 7)]),
]


def _build_items(seed, n):
    nf, ni = _rng(seed)
    items, true_violation = [], []
    for idx in range(n):
        cat = CAT_NAMES[ni(0, len(CAT_NAMES) - 1)]
        c = CATS[cat]
        p_true = c["base_rate"] + (nf() - 0.5) * 2 * 0.08
        p_true = min(0.97, max(0.03, p_true))
        tv = 1 if nf() < p_true else 0
        model_score = p_true + (nf() - 0.5) * 2 * c["noise"]
        model_score = min(0.98, max(0.02, model_score))
        ambiguity = c["ambig_base"] + (nf() - 0.5) * 2 * c["ambig_noise"]
        ambiguity = min(0.95, max(0.05, ambiguity))
        items.append({"id": idx, "category": cat,
                      "model_score": round(model_score, 4),
                      "ambiguity": round(ambiguity, 4)})
        true_violation.append(tv)
    return items, true_violation


def _build_instances():
    out = []
    for seed, n, rv_specs in SPECS:
        items, tv = _build_items(seed, n)
        reviewers = [{"id": i, "base_accuracy": ba, "fatigue_rate": fr, "capacity": cap}
                     for i, (ba, fr, cap) in enumerate(rv_specs)]
        inst = {
            "name": f"modq_{seed}",
            "n": n,
            "auto_threshold": AUTO_THRESHOLD,
            "floor_acc": FLOOR_ACC,
            "difficulty_penalty": DIFFICULTY_PENALTY,
            "categories": {k: {"severity": v["severity"], "appeal_cost": v["appeal_cost"]} for k, v in CATS.items()},
            "items": items,
            "reviewers": reviewers,
            "_true_violation": tv,
        }
        out.append(inst)
    return out


def _public_view(inst):
    return {k: v for k, v in inst.items() if not k.startswith("_")}


# ----------------------------- shared arithmetic -----------------------------
def eff_acc(base_acc, fatigue_rate, k, ambiguity):
    a = base_acc - fatigue_rate * k - DIFFICULTY_PENALTY * ambiguity
    return max(FLOOR_ACC, min(0.99, a))


def _realize_cost(inst, reviewed_of):
    """reviewed_of: item_id -> (reviewer_index, position). Deterministic (no sampling):
    every 'probability' is used as an expected-value weight, not a random draw."""
    items = inst["items"]; reviewers = inst["reviewers"]
    tv = inst["_true_violation"]; cats = inst["categories"]
    thr = inst["auto_threshold"]
    total = 0.0
    for i, it in enumerate(items):
        c = cats[it["category"]]
        loss = c["severity"] if tv[i] else c["appeal_cost"]
        if i in reviewed_of:
            rid, k = reviewed_of[i]
            rv = reviewers[rid]
            ea = eff_acc(rv["base_accuracy"], rv["fatigue_rate"], k, it["ambiguity"])
            total += (1.0 - ea) * loss
        else:
            removed = it["model_score"] >= thr
            if tv[i]:
                total += 0.0 if removed else c["severity"]
            else:
                total += c["appeal_cost"] if removed else 0.0
    return total


def _oracle_reviewed(inst):
    """Hidden-info reference: greedily spend capacity on the items with the largest
    TRUE benefit (uses the hidden true_violation label, unlike any candidate), routing
    each chosen item to whichever reviewer currently offers the best effective accuracy.
    A reasonable, computable UPPER anchor -- not claimed optimal."""
    items = inst["items"]; N = len(items); reviewers = inst["reviewers"]; R = len(reviewers)
    tv = inst["_true_violation"]; cats = inst["categories"]; thr = inst["auto_threshold"]

    loss, cost_auto = [], []
    for i, it in enumerate(items):
        c = cats[it["category"]]
        l = c["severity"] if tv[i] else c["appeal_cost"]
        loss.append(l)
        removed = it["model_score"] >= thr
        if tv[i]:
            ca = 0.0 if removed else c["severity"]
        else:
            ca = c["appeal_cost"] if removed else 0.0
        cost_auto.append(ca)

    def best_fresh_ea(i):
        amb = items[i]["ambiguity"]
        return max(eff_acc(rv["base_accuracy"], rv["fatigue_rate"], 0, amb) for rv in reviewers)

    order = sorted(range(N), key=lambda i: -(cost_auto[i] - (1.0 - best_fresh_ea(i)) * loss[i]))
    next_k = [0] * R
    cap_left = [rv["capacity"] for rv in reviewers]
    reviewed_of = {}
    for i in order:
        best_r, best_ea = -1, -1.0
        for r in range(R):
            if cap_left[r] <= 0:
                continue
            ea = eff_acc(reviewers[r]["base_accuracy"], reviewers[r]["fatigue_rate"], next_k[r], items[i]["ambiguity"])
            if ea > best_ea:
                best_ea, best_r = ea, r
        if best_r == -1:
            continue
        reviewed_cost = (1.0 - best_ea) * loss[i]
        if reviewed_cost < cost_auto[i] - 1e-12:
            reviewed_of[i] = (best_r, next_k[best_r])
            cap_left[best_r] -= 1
            next_k[best_r] += 1
    return reviewed_of


# ----------------------------- answer validation ------------------------------
def _validate(inst, answer):
    if not isinstance(answer, dict):
        return False, None
    schedule = answer.get("schedule")
    if not isinstance(schedule, dict):
        return False, None
    reviewers = inst["reviewers"]
    N = len(inst["items"])
    cap = {str(rv["id"]): rv["capacity"] for rv in reviewers}
    seen = set()
    reviewed_of = {}
    for key, lst in schedule.items():
        if not isinstance(key, str) or key not in cap:
            return False, None
        if not isinstance(lst, list):
            return False, None
        if len(lst) > cap[key]:
            return False, None
        for pos, item_id in enumerate(lst):
            if isinstance(item_id, bool) or not isinstance(item_id, int):
                return False, None
            if item_id < 0 or item_id >= N:
                return False, None
            if item_id in seen:
                return False, None
            seen.add(item_id)
            reviewed_of[item_id] = (int(key), pos)
    return True, reviewed_of


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        cost_auto = _realize_cost(inst, {})
        cost_oracle = _realize_cost(inst, _oracle_reviewed(inst))
        denom = max(1e-6, cost_auto - cost_oracle)

        public = _public_view(inst)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, reviewed_of = _validate(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        try:
            cost_cand = _realize_cost(inst, reviewed_of)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (cost_auto - cost_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
