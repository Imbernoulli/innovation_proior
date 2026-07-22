#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0842 -- "Edge Cache: Prefetching Along the Drifting Hot
Set on a Content Network".

Family: drift-anticipating-prefetch, skinned as an edge-cache node on a content
delivery network. A fixed catalog of M content items is served from a small edge
cache of capacity C. Over T discrete time steps, per-item request weight is given
directly in the instance (a T x M table): every item has an "evergreen" baseline
popularity, but a contiguous block of catalog ids is additionally HOT at any given
moment, and that hot block drifts deterministically across the catalog over time
(wrapping around, so ids that cooled off become hot again later). The candidate
chooses the cache CONTENTS at every time step (subject to the capacity cap) for the
whole horizon at once (it sees the full table up front -- there is no online replay).

THE TWO COMPOSED MECHANISMS:
  (1) eviction-refetch-cost-asymmetry -- holding an already-cached item costs
      NOTHING per step, but bringing a NEW item into the cache (whether that is
      the very first fetch or a re-fetch after an earlier eviction) always costs
      its full per-item fetch price again; meanwhile NOT having a requested item
      cached costs a miss penalty per unit of that step's request weight, and the
      miss penalty is set far above the cache-hit cost. So evicting something that
      will be wanted again soon is pure loss (you eventually pay to fetch it right
      back, on top of paying misses for however long it was missing).
  (2) popularity-locality-drift -- the hot block's position is a deterministic
      function of time baked into the weight table; nothing about it is random or
      wall-clock dependent, and it wraps around the catalog, so old locality
      recurs.

ONE-STEP FETCH LATENCY (ties the two mechanisms together): items newly entering
the cache at step t do NOT serve step t's own traffic -- only whatever was
ALREADY resident going into step t (i.e. the cache as it stood after step t-1)
serves step t. Anything added during step t only starts paying off from step t+1
onward. So avoiding a miss on an item that turns hot at step t REQUIRES having
already fetched it by the end of step t-1 -- genuine lookahead, not hindsight.

INNOVATION HOOK: the table already encodes tomorrow's hot set, so a solver that
looks AHEAD in the given table (at rows > t) and fetches an item BEFORE its hot
stretch begins amortizes one fetch cost against the whole stretch of misses it
prevents, and only evicts an item when a costlier-to-miss upcoming item needs the
slot. The TRAP is the classic online cache policy (LRU/LFU): even handed the full
table, it is natural to simulate a policy that only ever admits items based on
weight ALREADY observed (row t or earlier) -- which, combined with the fetch
latency above, means it is always chasing the drift from behind. It keeps paying
misses on the leading edge of the hot block and churns (evict, then refetch) on
items that cyclically return.

The candidate is UNTRUSTED model output: it runs in an ISOLATED subprocess via
`isorun`, sees ONLY the public instance on stdin, and returns ONLY its answer on
stdout, so it can never reach the evaluator's frames / scorer / baseline / hidden.

Scoring (deterministic; no wall-time):
  baseline b = cost of the best FIXED cache (chosen once, held for the whole
               horizon, by total weight across all T steps) -- the natural
               "ignore the drift" design. Always feasible.
  For a FEASIBLE answer with objective obj:  r = min(1, 0.1 * b / obj)
  Infeasible / malformed answer -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    def uf():  # uniform float in [0,1)
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    nxt.uf = uf
    return nxt


# ----------------------------- instance family -----------------------------
def _build_weight_table(M, T, K, stride, cold, hot_bump):
    """weight[t][i]: a small evergreen baseline `cold[i]` present every step, plus
    `hot_bump[i]` added on top while i sits in the length-K contiguous (circular)
    hot block that starts at (t*stride) mod M and slides forward by `stride` ids
    every step -- the deterministic drift. The bump is additive and independently
    scaled from the baseline so the (unavoidable, since C << M) background traffic
    on never-cached items stays a modest share of the objective -- the drift
    coverage decision, not baseline noise, is what dominates the score."""
    W = []
    for t in range(T):
        start = (t * stride) % M
        hotset = set((start + j) % M for j in range(K))
        row = [cold[i] + (hot_bump[i] if i in hotset else 0.0) for i in range(M)]
        W.append(row)
    return W


def make_instances():
    """Deterministic, seeded. Returns [{'public':..., 'hidden':{}}].

    Each instance: a catalog of M items, T time steps, cache capacity C. A hot
    block of K contiguous ids drifts by `stride` ids per step (wrapping around the
    catalog -- so ids cycle back into the hot block after M/stride steps). Every
    item has a baseline popularity `base[i]`, a per-unit-weight miss cost
    `miss_cost[i]` (>> the fixed hit cost of 1.0), and a flat re/fetch price
    `fetch_cost[i]` paid whenever the item newly enters the cache. miss_cost and
    fetch_cost share an item-level "size" multiplier, so heavier items are both
    costlier to miss and costlier to prefetch -- a real trade-off, not a free
    prefetch-everything shortcut. Capacity C sits close to K (a little above it),
    so the cache can hold roughly one hot block's worth plus a small margin: there
    is always room to prefetch the next few arriving ids ahead of time, but never
    room to just hoard the whole catalog -- eviction timing genuinely matters.
    trap cases (large stride relative to K, and/or T spanning >=2 full wraps of
    the catalog) are marked; on those the reactive (lag-1) policy pays repeated
    leading-edge misses AND repeated churn on ids that cyclically return."""
    specs = [
        # seed, M,  T,  K,  stride, C,    bump, miss_lo,miss_hi, fetch_lo,fetch_hi, trap
        (311, 40, 24,  8,   2,      10,   14.0,     6, 10,   45, 90,   False),
        (312, 48, 28,  9,   3,      12,   16.0,     6, 10,   45, 90,   True),
        (313, 36, 30,  7,   4,      9,    18.0,     7, 11,   40, 85,   True),
        (314, 50, 26,  10,  2,      13,   13.0,     6, 9,    50, 95,   False),
        (315, 44, 32,  8,   3,      11,   17.0,     7, 12,   40, 90,   True),
        (316, 55, 30,  11,  2,      14,   14.0,     6, 10,   50, 100,  False),
        (317, 42, 34,  8,   4,      10,   19.0,     8, 13,   35, 80,   True),
        # larger / held-out instances (deeper drift, tighter margins)
        (318, 64, 40,  12,  3,      15,   16.0,     7, 12,   45, 95,   True),
        (319, 70, 36,  13,  4,      16,   18.0,     8, 13,   40, 90,   True),
        (320, 60, 44,  10,  3,      12,   20.0,     8, 14,   35, 85,   True),
    ]
    out = []
    for seed, M, T, K, stride, C, bump, miss_lo, miss_hi, fetch_lo, fetch_hi, trap in specs:
        r = _rng(seed)
        # small evergreen baseline present every step (kept modest so the
        # unavoidable background-traffic floor doesn't drown out the
        # hot/cold coverage decision the mechanisms are built around)
        cold = [0.02 + r.uf() * 0.05 for _ in range(M)]          # in [0.02,0.07)
        hot_bump = [1.7 * bump * (0.8 + r.uf() * 0.5) for _ in range(M)]  # in [1.36,2.21)*bump
        size_mult = [0.7 + r.uf() * 0.8 for _ in range(M)]      # item "size" in [0.7,1.5)
        miss_cost = [round((miss_lo + r.uf() * (miss_hi - miss_lo)) * size_mult[i], 4) for i in range(M)]
        fetch_cost = [round((fetch_lo + r.uf() * (fetch_hi - fetch_lo)) * size_mult[i], 4) for i in range(M)]
        W = _build_weight_table(M, T, K, stride, cold, hot_bump)
        public = {
            "M": M, "T": T, "C": C,
            "hit_cost": 1.0,
            "miss_cost": miss_cost,
            "fetch_cost": fetch_cost,
            "weight": [[round(w, 5) for w in row] for row in W],
        }
        out.append({"public": public, "hidden": {"trap": trap, "K": K, "stride": stride}})
    return out


# ----------------------------- scoring -------------------------------------
def _simulate(inst, cache_seq):
    """cache_seq: list of T sets of item ids (already validated). Returns total cost.

    ONE-STEP FETCH LATENCY: step t's traffic is served by `prev` -- the cache as it
    stood AFTER step t-1's changes (empty before step 0) -- NOT by cache_seq[t]
    itself. Items newly entering cache_seq[t] pay their fetch cost immediately but
    only start serving traffic from step t+1 onward. So protecting step t requires
    the item to already be resident in cache_seq[t-1]."""
    pub = inst["public"]
    M = pub["M"]; T = pub["T"]; hit_cost = pub["hit_cost"]
    miss_cost = pub["miss_cost"]; fetch_cost = pub["fetch_cost"]; W = pub["weight"]
    total = 0.0
    prev = set()
    for t in range(T):
        cur = cache_seq[t]
        for i in (cur - prev):
            total += fetch_cost[i]
        row = W[t]
        for i in range(M):
            w = row[i]
            if w <= 0.0:
                continue
            total += (hit_cost * w) if i in prev else (miss_cost[i] * w)
        prev = cur
    return total


def baseline(inst):
    """Cost of the TRUE-OPTIMAL FIXED cache (chosen once, held for all T steps):
    the best possible 'ignore the drift' non-adaptive design. A fixed set S is
    resident for steps 1..T-1 (step 0 is always served by the empty pre-cache, by
    construction, regardless of S -- see _simulate's one-step fetch latency) and
    pays each member's fetch_cost exactly once (at t=0). Because cost(S) is a
    CONSTANT plus a sum of independent per-item terms (each item occupies exactly
    one slot, no cross-item interaction beyond the shared capacity C), the optimal
    S is exactly the top-C items by gain[i], keeping only strictly positive gains:
        gain[i] = (miss_cost[i] - hit_cost) * sum_{t=1}^{T-1} weight[t][i] - fetch_cost[i]
    (the miss cost saved by holding i for steps 1..T-1, net of its one-time fetch
    price). This is an exact optimum over ALL fixed caches, not a heuristic, so no
    cost-aware-but-still-non-adaptive submission can beat this baseline's ratio."""
    pub = inst["public"]
    M = pub["M"]; T = pub["T"]; C = pub["C"]; W = pub["weight"]
    hit_cost = pub["hit_cost"]; miss_cost = pub["miss_cost"]; fetch_cost = pub["fetch_cost"]
    tot_w_from1 = [0.0] * M
    for t in range(1, T):
        row = W[t]
        for i in range(M):
            tot_w_from1[i] += row[i]
    gain = [(miss_cost[i] - hit_cost) * tot_w_from1[i] - fetch_cost[i] for i in range(M)]
    order = sorted(range(M), key=lambda i: (-gain[i], i))
    fixed = set(i for i in order[:C] if gain[i] > 0.0)
    cache_seq = [fixed for _ in range(T)]
    return _simulate(inst, cache_seq)


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    pub = inst["public"]
    M = pub["M"]; T = pub["T"]; C = pub["C"]
    if not isinstance(answer, dict):
        return False, None
    cache = answer.get("cache", None)
    if not isinstance(cache, list) or len(cache) != T:
        return False, None
    cache_seq = []
    for row in cache:
        if not isinstance(row, list) or len(row) > C:
            return False, None
        seen = set()
        for x in row:
            if not isinstance(x, int) or isinstance(x, bool):
                return False, None
            if x < 0 or x >= M:
                return False, None
            if x in seen:
                return False, None
            seen.add(x)
        cache_seq.append(seen)
    obj = _simulate(inst, cache_seq)
    if not math.isfinite(obj) or obj <= 0.0:
        return False, None
    return True, float(obj)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None or obj <= 0:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
