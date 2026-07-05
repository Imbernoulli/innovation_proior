#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0332 -- "Sump Expedition: Porter Load Planning"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  A cave-mapping expedition must haul all of its survey gear down a single
narrow shaft to the base camp at the bottom.  Gear travels in PORTER LOADS -- one
load is one trip down and back up the shaft.  Every load is subject to TWO limits:

  (1) a WEIGHT limit C: the total weight of the gear in one load must not exceed C
      (the rigging can only take so much on a single descent);
  (2) a HANDLING limit K: a single load may mix at most K DISTINCT gear CATEGORIES
      (ropes, carbide lamps, dye tracers, water samples, ...).  Each category needs
      its own dedicated stowage and cross-contamination protocol, and a porter can
      only manage K of those protocols on one descent.  There is NO limit on how
      many items of the SAME category ride together (beyond the weight limit).

Every trip down the shaft is expensive, so the expedition wants to move ALL the
gear using as FEW porter loads as possible.

This is CLASS-CONSTRAINED 1-D bin packing skinned as a caving expedition: gear
items = items with an integer weight AND a category label; a porter load = a bin
with a capacity limit C AND a distinct-class limit K; "loads dispatched" = bins
used, which we MINIMIZE.  The extra class limit is what makes it a fresh variant:
a load can be far below its weight limit yet still be "full" because it already
mixes K categories, so a good planner must think about which categories share a
load, not just how heavy each load is.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "capacity": C (int),         # per-load weight limit
             "classes": K (int),          # per-load distinct-category limit
             "n": N (int),                # number of gear items
             "weights": [w_0, ..., w_{N-1}],     # integer, 1 <= w_i <= C
             "category": [c_0, ..., c_{N-1}]}    # integer category id, 0 <= c_i
  stdout: ONE JSON object:
            {"assign": [b_0, ..., b_{N-1}]}
          where b_i >= 0 is the porter-load index that gear item i rides in.  Load
          indices need not be contiguous; a load "exists" iff >=1 item rides it, and
          the number of DISTINCT non-empty loads is the trip count.

  A plan is VALID iff `assign` is a list of exactly N non-negative integers and, for
  every load, the total weight <= C AND the number of distinct categories <= K.
  Invalid output, wrong length, an over-weight load, a load mixing more than K
  categories, a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    q_lb   = max( ceil(sum(weights)/C),  ceil(D/K) )      # unreachable ideal
             where D = number of DISTINCT categories present in the instance.
             (Both terms are valid lower bounds: total weight forces >= W/C loads;
             and with <= K classes per load, B loads cover at most B*K class-slots,
             so B >= D/K.)
    q_base = loads used by the internal NEXT-FIT operator                # weak baseline
    q_cand = loads used by the candidate plan
  and normalize with an affine anchor (weak baseline -> 0.1, ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate matching next-fit scores ~0.1; a candidate reaching the (generally
  unreachable) two-term lower bound scores 1.0; doing worse than next-fit scores
  < 0.1.  Because the lower bound is LOOSE, even strong class-aware packers stay
  strictly below 1.0 on most instances -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(lower bound, next-fit baseline) are computed by THIS parent process, so a
frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _build_items(seed, n, C, M, dist):
    """Return (weights, category), each a list of length n. Deterministic.
    weights in [1, C]; category in [0, M-1]."""
    ni = _rng(seed)
    weights = []
    category = []
    for _ in range(n):
        if dist == "uni":
            w = ni(1, C)
        elif dist == "light":                        # many small items; class limit dominates
            w = ni(1, max(1, C // 3))
        elif dist == "medium":                       # near half a load
            w = ni(max(1, C // 4), (3 * C) // 4)
        elif dist == "heavy":                        # mostly heavy items, hard to pair
            w = ni(max(1, (2 * C) // 5), (17 * C) // 20)
        else:
            w = ni(1, C)
        if w < 1:
            w = 1
        if w > C:
            w = C
        weights.append(w)
        category.append(ni(0, M - 1))
    return weights, category


def _build_instances():
    """Deterministic instance family. (seed, n, C, K, M, dist)."""
    specs = [
        (126, 24, 18, 2, 4, "heavy"),
        (122, 32, 22, 2, 5, "heavy"),
        (104, 28, 18, 2, 6, "heavy"),
        (108, 28, 24, 2, 5, "heavy"),
        (121, 32, 24, 3, 6, "medium"),
        (135, 28, 20, 3, 4, "medium"),
        (137, 32, 22, 3, 6, "uni"),
        (133, 28, 18, 3, 7, "medium"),
        (125, 32, 20, 2, 6, "uni"),
        (115, 32, 20, 2, 5, "medium"),
        # harder / larger held-out instances
        (106, 46, 24, 2, 8, "heavy"),
        (226, 40, 22, 2, 5, "uni"),
    ]
    out = []
    for seed, n, C, K, M, dist in specs:
        weights, category = _build_items(seed, n, C, M, dist)
        out.append({"name": f"expd{seed}", "capacity": C, "classes": K,
                    "n": n, "weights": weights, "category": category})
    return out


# ----------------------------- references ----------------------------------
def _lower_bound(weights, category, C, K):
    w_term = -(-sum(weights) // C)                    # ceil(sum / C)
    D = len(set(category))
    c_term = -(-D // K)                               # ceil(D / K)
    return max(w_term, c_term)


def _next_fit(weights, category, C, K):
    """Weak online operator: keep loading the current porter load; open a NEW load
    the moment the next item would break EITHER the weight limit OR the K-category
    limit.  Never looks back."""
    loads = 1
    rem = C
    cats = set()
    for w, c in zip(weights, category):
        fits_w = (w <= rem)
        fits_c = (c in cats) or (len(cats) < K)
        if fits_w and fits_c:
            rem -= w
            cats.add(c)
        else:
            loads += 1
            rem = C - w
            cats = {c}
    return loads


# ----------------------------- validation ----------------------------------
def _loads_used(inst, answer):
    """Validate answer against the instance. Return load count or None."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    weights = inst["weights"]
    category = inst["category"]
    C = inst["capacity"]
    K = inst["classes"]
    N = inst["n"]
    if len(assign) != N:
        return None
    load_w = {}
    load_c = {}
    for i, b in enumerate(assign):
        if isinstance(b, bool) or not isinstance(b, int):
            return None
        if b < 0:
            return None
        load_w[b] = load_w.get(b, 0) + weights[i]
        if load_w[b] > C:
            return None
        s = load_c.get(b)
        if s is None:
            s = set()
            load_c[b] = s
        s.add(category[i])
        if len(s) > K:
            return None
    return len(load_w)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        C = inst["capacity"]
        K = inst["classes"]
        weights = inst["weights"]
        category = inst["category"]
        q_lb = _lower_bound(weights, category, C, K)
        q_base = _next_fit(weights, category, C, K)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": C, "classes": K,
                  "n": inst["n"], "weights": list(weights),
                  "category": list(category)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _loads_used(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_base - q_cand) / denom
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
