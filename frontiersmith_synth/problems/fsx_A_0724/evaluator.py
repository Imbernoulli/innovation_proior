#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0724 -- "Water-Filling a Measurement Budget Across Coupled
Quantities" (family: measurement-budget-water-filling; format B, quality-metric).

THEME.  A lab must estimate Q unknown quantities before a deadline. It has M possible
MEASUREMENT TYPES it can run, each with an integer "how-many-times" dial n_m in
[0, cap_m], costing exactly 1 budget unit per run, total runs <= budget B. Running
measurement m reduces the POSTERIOR VARIANCE of every quantity it is informative about
-- some measurements are private probes of a single quantity, others are SHARED probes
whose single run is simultaneously informative about several quantities at once (a
"hidden" cross-coupling given explicitly as a coverage matrix). Each measurement has a
concave, saturating cost-to-precision curve (diminishing returns per additional run).
The instance also gives each quantity a PRIOR precision (information already known
before any measurement). The candidate must output one integer allocation; it is graded
on the worst (largest) POSTERIOR VARIANCE across all Q quantities after applying every
funded measurement run.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"instance_id": str, "n_quantities": Q, "n_measurements": M, "budget": B,
             "prior_precision": [p_0, ..., p_{Q-1}],
             "gain_a": [a_0, ..., a_{M-1}], "gain_b": [b_0, ..., b_{M-1}],
             "cap": [cap_0, ..., cap_{M-1}],
             "coverage": [[c_{m,0}, ..., c_{m,Q-1}] for m in range(M)]}
  stdout: ONE JSON object: {"alloc": [n_0, ..., n_{M-1}]}
          n_m: non-negative INTEGER number of runs of measurement m.

  A valid answer has "alloc" a list of exactly M non-negative integers, n_m <= cap_m,
  and sum(n_m) <= B. Any violation, a crash, a timeout, or non-JSON output makes that
  instance score 0.0.

MECHANICS (identical for every instance, computed by the PARENT, never sent to the
candidate as a formula -- only the raw numbers are given).
  Each measurement m has a concave "information-gain-probe" curve
      gain_m(n) = a_m * n / (n + b_m)          (gain_m(0) = 0, increasing, saturating)
  Running an allocation `alloc` gives quantity q total posterior INFORMATION
      info_q(alloc) = prior_precision[q] + sum_m coverage[m][q] * gain_m(alloc[m])
  and posterior VARIANCE var_q(alloc) = 1 / info_q(alloc). A measurement with
  coverage[m][q] > 0 for several q is a SHARED probe: ONE run helps every quantity it
  covers simultaneously -- but only counts as ONE unit of budget, so a well-designed
  allocation should recognise (and fund) that cross-coverage rather than paying
  separately, per quantity, for redundant private probes.

SCORING.  Per instance, minimize the WORST posterior variance max_q var_q(alloc). The
evaluator computes, itself, two references never revealed to the candidate: obj_base
(a naive even-split-the-budget construction) and obj_ref (a stronger internal
water-filling + integer-repair + local-swap-polish procedure -- a strong but not
provably-optimal ceiling). Then, with obj_base > obj_ref (both are variances; lower is
better so obj_base is the worse/larger number):
    r = clamp(0.1 + 0.9 * (obj_base - obj_cand) / max(1e-9, 1.5*(obj_base - obj_ref)), 0, 1)
Matching the naive baseline scores ~0.1; matching the strong internal reference scores
~0.7; there is headroom above that (the internal reference is a heuristic, not a proven
optimum) and below the 0.1 floor is impossible once feasible (r is clamped at 0).
The reported Ratio is the mean r over 10 fixed instances; Vector lists the 10 scores.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance. obj_base/obj_ref are
computed by THIS parent process, so a frame-walking / introspecting candidate learns
nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- gain curve -----------------------------------
def _gain(a, b, n):
    if n <= 0:
        return 0.0
    return a * n / (n + b)


def _dgain(a, b, n):
    # marginal gain of the (n+1)-th unit, n >= 0
    return _gain(a, b, n + 1) - _gain(a, b, n)


# ------------------------- instance construction ----------------------------
def _build_group(prior_list, priv_specs, shared_spec, rng):
    """
    prior_list  : list of prior_precision, one per quantity in this group.
    priv_specs  : list of (a, b, cap) private-measurement params, one per quantity
                  (same length as prior_list); a private measurement covers ONLY its
                  own quantity (coverage weight 1.0).
    shared_spec : (a, b, cap) for ONE measurement covering every quantity in the group
                  with coverage weight 1.0, or None if the group has no shared probe.
    Returns (priors, measurements) where measurements is a list of
      {"a":a, "b":b, "cap":cap, "cov": {local_q_index: weight}}
    with local indices 0..len(prior_list)-1 (caller offsets into the global arrays).
    """
    gs = len(prior_list)
    measurements = []
    for qi, (a, b, cap) in enumerate(priv_specs):
        measurements.append({"a": a, "b": b, "cap": cap, "cov": {qi: 1.0}})
    if shared_spec is not None and gs > 1:
        a, b, cap = shared_spec
        measurements.append({"a": a, "b": b, "cap": cap, "cov": {qi: 1.0 for qi in range(gs)}})
    return list(prior_list), measurements


def _assemble(name, seed, groups, B, n_distractors=0):
    """groups: list of (prior_list, priv_specs, shared_spec)."""
    rng = _rng(seed)
    priors = []
    meas = []  # list of {"a","b","cap","cov": {global_q: weight}}
    off = 0
    for prior_list, priv_specs, shared_spec in groups:
        gpriors, gmeas = _build_group(prior_list, priv_specs, shared_spec, rng)
        priors.extend(gpriors)
        for m in gmeas:
            meas.append({"a": m["a"], "b": m["b"], "cap": m["cap"],
                         "cov": {off + k: v for k, v in m["cov"].items()}})
        off += len(prior_list)
    Q = off
    # a handful of low-value distractor measurements: cheap-looking (small b) but tiny
    # asymptote and touching a single, already-adequate quantity picked at random --
    # noise a sound solver should simply not bother funding much.
    for _ in range(n_distractors):
        q = rng(0, Q - 1)
        a = rng(1, 2) * 0.6
        b = 0.3 + rng(0, 3) * 0.1
        cap = rng(2, 4)
        meas.append({"a": a, "b": b, "cap": cap, "cov": {q: 1.0}})
    M = len(meas)
    coverage = [[0.0] * Q for _ in range(M)]
    gain_a = [0.0] * M
    gain_b = [0.0] * M
    cap = [0] * M
    for m, spec in enumerate(meas):
        gain_a[m] = float(spec["a"])
        gain_b[m] = float(spec["b"])
        cap[m] = int(spec["cap"])
        for q, w in spec["cov"].items():
            coverage[m][q] = float(w)
    return {"name": name, "n_quantities": Q, "n_measurements": M, "budget": B,
            "prior_precision": priors, "gain_a": gain_a, "gain_b": gain_b,
            "cap": cap, "coverage": coverage}


def _build_instances():
    out = []

    # -- 0,1,2: calm warm-up instances -- moderate trap, single 2-quantity group,
    #    a shared probe that is genuinely the efficient choice.
    out.append(_assemble("m0", 101, [
        ([1.2, 1.0], [(4.0, 1.2, 5), (4.0, 1.4, 5)], (18.0, 7.0, 24)),
    ], B=22, n_distractors=1))
    out.append(_assemble("m1", 102, [
        ([0.9, 1.1], [(5.0, 1.0, 5), (4.5, 1.3, 5)], (20.0, 7.5, 26)),
    ], B=24, n_distractors=1))
    out.append(_assemble("m2", 103, [
        ([1.5, 0.8, 1.0], [(4.0, 1.1, 4), (4.5, 1.0, 4), (4.0, 1.3, 4)], (22.0, 8.0, 26)),
    ], B=26, n_distractors=2))

    # -- 3..7: TRAP instances -- one "already pinned" quantity (high prior, deceptively
    #    steep tiny-b private probe) paired with one or two truly "needy" quantities
    #    (low prior, weaker private slope) in the SAME group, sharing one probe. The
    #    static per-quantity-cheapest-first recipe funds the pinned quantity's private
    #    probe first (steepest initial slope) even though it barely moves the worst
    #    case, starving the needy quantities / the efficient shared probe.
    out.append(_assemble("t3", 201, [
        ([9.0, 0.35], [(3.0, 0.35, 4), (2.6, 2.6, 4)], (18.0, 6.5, 22)),
    ], B=20, n_distractors=1))
    out.append(_assemble("t4", 202, [
        ([10.0, 0.30, 0.40], [(3.2, 0.3, 4), (2.8, 3.0, 4), (2.6, 3.2, 4)], (24.0, 7.0, 28)),
    ], B=24, n_distractors=2))
    out.append(_assemble("t5", 203, [
        ([8.5, 0.32], [(3.4, 0.28, 5), (2.4, 2.8, 4)], (20.0, 6.0, 24)),
        ([0.6], [(3.0, 1.6, 5)], None),
    ], B=22, n_distractors=1))
    out.append(_assemble("t6", 204, [
        ([11.0, 0.28, 0.35], [(3.6, 0.25, 5), (2.6, 3.0, 4), (2.5, 3.2, 4)], (26.0, 7.5, 30)),
    ], B=26, n_distractors=2))
    out.append(_assemble("t7", 205, [
        ([9.5, 0.30], [(3.3, 0.3, 4), (2.5, 2.9, 4)], (19.0, 6.2, 22)),
        ([9.0, 0.33], [(3.1, 0.32, 4), (2.4, 2.7, 4)], (19.0, 6.4, 22)),
    ], B=28, n_distractors=2))

    # -- 8,9: harder / held-out -- three groups, tighter relative budget, bigger Q/M.
    out.append(_assemble("h8", 301, [
        ([9.5, 0.30, 0.40], [(3.4, 0.28, 4), (2.6, 2.8, 4), (2.5, 3.0, 4)], (24.0, 7.2, 26)),
        ([0.7, 0.6], [(3.0, 1.6, 4), (2.8, 1.8, 4)], (16.0, 6.0, 20)),
    ], B=30, n_distractors=2))
    out.append(_assemble("h9", 302, [
        ([10.5, 0.28, 0.35], [(3.5, 0.25, 4), (2.7, 2.9, 4), (2.5, 3.1, 4)], (26.0, 7.5, 26)),
        ([0.65, 0.55], [(2.9, 1.7, 4), (2.7, 1.9, 4)], (15.0, 6.2, 18)),
        ([0.5], [(2.6, 1.9, 4)], None),
    ], B=30, n_distractors=2))
    return out


# --------------------------------- objective ---------------------------------
def _infos(inst, alloc):
    Q = inst["n_quantities"]
    M = inst["n_measurements"]
    gains = [_gain(inst["gain_a"][m], inst["gain_b"][m], alloc[m]) for m in range(M)]
    infos = list(inst["prior_precision"])
    for m in range(M):
        g = gains[m]
        if g == 0.0:
            continue
        row = inst["coverage"][m]
        for q in range(Q):
            w = row[q]
            if w:
                infos[q] += w * g
    return infos


def _worst_var(inst, alloc):
    infos = _infos(inst, alloc)
    return max(1.0 / i for i in infos)


# --------------------------- reference constructions --------------------------
def _baseline_alloc(inst):
    M = inst["n_measurements"]
    B = inst["budget"]
    n_each, rem = divmod(B, M)
    alloc = [n_each] * M
    for i in range(rem):
        alloc[i] += 1
    for m in range(M):
        if alloc[m] > inst["cap"][m]:
            alloc[m] = inst["cap"][m]
    return alloc


def _dynamic_waterfill(inst):
    """Stage 1 reference procedure (also what the 'strong' candidate implements):
    integer water-filling, one budget unit at a time. Each unit is committed --
    rounded straight to an integer, never left fractional -- to whichever available
    measurement gives the CURRENT worst quantity (real posterior info, prior
    included) the largest marginal information gain; the worst quantity, and every
    measurement's up-to-date marginal gain, are recomputed after every single unit,
    so a measurement's cross-coverage of several quantities is automatically
    'funded once' and its benefit is felt by all of them without ever being
    double-booked. If no measurement still covers the current worst quantity (all
    capped), fall back to the measurement whose next unit yields the largest total
    marginal VARIANCE reduction summed over every quantity it covers (weighting each
    quantity's marginal info gain by 1/info_q^2, the exact derivative of variance),
    so budget is never simply wasted."""
    Q = inst["n_quantities"]
    M = inst["n_measurements"]
    B = inst["budget"]
    a = inst["gain_a"]; b = inst["gain_b"]; cap = inst["cap"]
    cov = inst["coverage"]

    alloc = [0] * M
    for _ in range(B):
        infos = _infos(inst, alloc)
        worst_q = min(range(Q), key=lambda q: infos[q])
        best_m, best_delta = None, -1.0
        for m in range(M):
            if alloc[m] >= cap[m]:
                continue
            w = cov[m][worst_q]
            if w <= 0:
                continue
            delta = w * _dgain(a[m], b[m], alloc[m])
            if delta > best_delta:
                best_delta = delta
                best_m = m
        if best_m is None:
            best_val = -1.0
            for m in range(M):
                if alloc[m] >= cap[m]:
                    continue
                val = 0.0
                for q in range(Q):
                    wq = cov[m][q]
                    if wq:
                        val += wq * _dgain(a[m], b[m], alloc[m]) / (infos[q] ** 2)
                if val > best_val:
                    best_val = val
                    best_m = m
        if best_m is None:
            break
        alloc[best_m] += 1
    return alloc


def _swap_polish(inst, alloc, passes=3):
    M = inst["n_measurements"]
    cap = inst["cap"]
    cur = list(alloc)
    cur_obj = _worst_var(inst, cur)
    for _ in range(passes):
        improved = False
        for mf in range(M):
            for mt in range(M):
                if mt == mf or cur[mf] <= 0 or cur[mt] >= cap[mt]:
                    continue
                trial = list(cur)
                trial[mf] -= 1
                trial[mt] += 1
                tv = _worst_var(inst, trial)
                if tv < cur_obj - 1e-12:
                    cur = trial
                    cur_obj = tv
                    improved = True
        if not improved:
            break
    return cur


def _internal_reference(inst):
    alloc = _dynamic_waterfill(inst)
    alloc = _swap_polish(inst, alloc, passes=3)
    return alloc


# ----------------------------- answer validation -----------------------------
def _validate_answer(answer, M, cap, B):
    if not isinstance(answer, dict):
        return None
    alloc = answer.get("alloc")
    if not isinstance(alloc, list) or len(alloc) != M:
        return None
    total = 0
    out = []
    for m, x in enumerate(alloc):
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        if x < 0 or x > cap[m]:
            return None
        out.append(x)
        total += x
    if total > B:
        return None
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        Q = inst["n_quantities"]; M = inst["n_measurements"]; B = inst["budget"]
        cap = inst["cap"]

        base_alloc = _baseline_alloc(inst)
        obj_base = _worst_var(inst, base_alloc)
        ref_alloc = _internal_reference(inst)
        obj_ref = _worst_var(inst, ref_alloc)
        denom = 1.5 * (obj_base - obj_ref)
        if denom < 1e-9:
            denom = 1e-9

        public = {"instance_id": inst["name"], "n_quantities": Q, "n_measurements": M,
                  "budget": B, "prior_precision": list(inst["prior_precision"]),
                  "gain_a": list(inst["gain_a"]), "gain_b": list(inst["gain_b"]),
                  "cap": list(cap), "coverage": [list(row) for row in inst["coverage"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            alloc = _validate_answer(ans, M, cap, B)
        except Exception:
            alloc = None
        if alloc is None:
            vec.append(0.0)
            continue
        try:
            obj_cand = _worst_var(inst, alloc)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (obj_base - obj_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
