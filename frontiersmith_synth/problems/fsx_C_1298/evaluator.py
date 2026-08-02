#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1298 -- "Ants That Must Not All Follow the Same Trail"
(family: colony-forage-policy; format B, quality-metric).

THEME.  A forager colony sends A ants out every round across K food SOURCES.  A
source's usable output is NOT driven by how many ants you send it *this instant*
-- it is driven by the colony's accumulated PHEROMONE TRAIL to that source, which
only catches up to a new allocation gradually (real ant trails build up over many
successive trips and evaporate when abandoned):

    trail_i(t) = decay_i * trail_i(t-1) + (1 - decay_i) * a_i(t)
    potential_i(t) = rate_i * trail_i(t)
    harvest_i(t)  = min(stock_i(t), potential_i(t))
    stock_i(t+1)  = min(cap_i, stock_i(t) - harvest_i(t) + regen_i)

Every source is a finite, regenerating larder: `stock_i` depletes when harvested
and refills by `regen_i` per round, capped at `cap_i`.  The colony wants to
MAXIMIZE total food harvested over T rounds, with a fixed budget of A ants to
place across the K sources each round.

WHY THIS IS NOT A ONE-SHOT RANKING PROBLEM.  Sending every ant to the source
with the best raw per-ant rate looks optimal round after round -- it IS optimal
right up until that source's one-time surplus stock is drained to its regen
floor.  From that point every extra ant camped there is wasted (harvest is
capped by `stock`/`regen`, not by trail), while a source that was never fed has
ZERO trail and needs several rounds of investment before it can absorb ants at
all (the `decay_i` lag).  A policy that never re-diversifies -- reinforcing the
best-known trail forever -- locks the colony onto a depleting source and starves
once it dries up, precisely because it did nothing to pre-warm anywhere else.
The insight this problem rewards is DECAYING the reinforcement on a source at a
rate that tracks its own depletion, so the colony's trail-building investment on
the next source lands exactly when the current one runs dry -- not before (wasted
warm-up) and not after (cold-start drought).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "K": int, "T": int, "A": int,
             "sources": [{"stock0": int, "cap": int, "regen": int,
                          "rate": float, "decay": float}, ...]}   # length K
  stdout: ONE JSON object:
            {"alloc": [[a_0_0, ..., a_0_{K-1}], ..., T rows]}
          `alloc` must have exactly T rows of exactly K non-negative integers
          each; each row's sum must not exceed A (idle ants are allowed but do
          nothing).  Wrong shape, a non-integer/negative/bool entry, a row
          summing above A, a crash, a timeout, or non-JSON output -> that
          instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute, in THIS parent
process (never sent to the candidate):
    base = total harvest of an EQUAL, never-adapting split of A ants across the
           K sources every round (the "do nothing clever" reference),
    ub   = 1.08 * (best total harvest found by a generic seeded local search --
           NOT the phased/decaying-reinforcement strategy this problem is meant
           to teach -- started from a few plain policies and refined by many
           random single-ant transfers), an unreached, loosely optimistic ceiling.
  and normalize with an affine anchor (equal-split -> 0.1, near-ceiling -> ~1.0):
    r = clamp( 0.1 + 0.9 * (harvest_cand - base) / max(1e-9, ub - base), 0, 1 )
  A candidate matching the equal split scores ~0.1; a candidate reaching the
  generic-search ceiling scores close to 1.0 (never exactly, by the 1.08 margin);
  doing worse than equal split scores below 0.1.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  `base`/`ub` and
the source dynamics are computed by THIS parent process, so a frame-walking /
introspecting candidate learns nothing useful.

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

    def ni(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 33) % (hi - lo + 1)

    def nf():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return ni, nf


# ----------------------------- instance family -----------------------------
def _make_sources(seed, K, kind):
    """Deterministic K sources. kind='abundant' (generous, no real depletion
    within the horizon) or 'patchy' (one high-rate 'jackpot' source with a
    small one-time stock and a tiny regen -- the depletion trap; the rest have
    lower rate but ample stock/regen)."""
    ni, nf = _rng(seed)
    srcs = []
    for k in range(K):
        if kind == "abundant":
            rate = 0.6 + 0.5 * nf()
            stock0 = ni(900, 1400)
            cap = stock0 + ni(200, 400)
            regen = ni(45, 70)
            decay = 0.75 + 0.15 * nf()
        else:
            if k == 0:                                    # the jackpot trap
                rate = 1.5 + 0.3 * nf()
                stock0 = ni(420, 560)
                cap = stock0 + ni(40, 90)
                regen = ni(8, 16)
            else:
                rate = 0.5 + 0.5 * nf()
                stock0 = ni(500, 900)
                cap = stock0 + ni(150, 300)
                regen = ni(28, 46)
            decay = 0.80 + 0.13 * nf()
        srcs.append({"stock0": stock0, "cap": cap, "regen": regen,
                     "rate": round(rate, 3), "decay": round(decay, 3)})
    return srcs


# specs: (name_seed, K, kind, T, A) -- 3 abundant warm-ups + 7 patchy traps,
# with varied horizon/colony-size for generalization (a few larger held-out).
_SPECS = [
    (3101, 4, "abundant", 20, 55),
    (3102, 5, "abundant", 22, 60),
    (3103, 3, "abundant", 24, 65),
    (3201, 4, "patchy",   20, 55),
    (3202, 5, "patchy",   22, 60),
    (3203, 3, "patchy",   18, 50),
    (3204, 4, "patchy",   24, 65),
    (3205, 5, "patchy",   26, 70),   # held-out: longer horizon
    (3206, 6, "patchy",   22, 60),   # held-out: more sources
    (3207, 4, "patchy",   30, 75),   # held-out: longest horizon + most ants
]


def _build_instances():
    out = []
    for seed, K, kind, T, A in _SPECS:
        srcs = _make_sources(seed, K, kind)
        out.append({"name": f"colony{seed}", "K": K, "T": T, "A": A,
                    "sources": srcs, "kind": kind})
    return out


# ----------------------------- simulator ------------------------------------
def _simulate(srcs, T, A, alloc):
    """alloc: T rows of K non-negative ints (already validated). Returns total
    harvested food (float)."""
    K = len(srcs)
    stock = [s["stock0"] for s in srcs]
    trail = [0.0] * K
    total = 0.0
    for t in range(T):
        row = alloc[t]
        for i in range(K):
            s = srcs[i]
            trail[i] = s["decay"] * trail[i] + (1 - s["decay"]) * row[i]
            potential = s["rate"] * trail[i]
            h = stock[i] if stock[i] < potential else potential
            total += h
            nxt = stock[i] - h + s["regen"]
            stock[i] = s["cap"] if nxt > s["cap"] else nxt
    return total


def _uniform_alloc(K, T, A):
    base = A // K
    rem = A - base * K
    row = [base] * K
    for i in range(rem):
        row[i] += 1
    return [row[:] for _ in range(T)]


def _rate_prop_alloc(srcs, K, T, A):
    w = [s["rate"] for s in srcs]
    wsum = sum(w)
    row_f = [A * wi / wsum for wi in w]
    row = [int(x) for x in row_f]           # floor (rates are positive)
    rem = A - sum(row)
    order = sorted(range(K), key=lambda i: -(row_f[i] - row[i]))
    for i in range(rem):
        row[order[i % K]] += 1
    return [row[:] for _ in range(T)]


def _myopic_alloc(srcs, K, T, A):
    """All-in each round on whichever source projects the best ONE-STEP
    harvest given trail built so far -- a plausible online rule, used only to
    seed the reference search (never a candidate tier)."""
    stock = [s["stock0"] for s in srcs]
    trail = [0.0] * K
    alloc = []
    for t in range(T):
        best, bestval = -1, -1.0
        for i in range(K):
            s = srcs[i]
            proj_trail = s["decay"] * trail[i] + (1 - s["decay"]) * A
            proj = stock[i] if stock[i] < s["rate"] * proj_trail else s["rate"] * proj_trail
            if proj > bestval:
                bestval, best = proj, i
        row = [0] * K
        row[best] = A
        alloc.append(row)
        for i in range(K):
            s = srcs[i]
            trail[i] = s["decay"] * trail[i] + (1 - s["decay"]) * row[i]
            h = stock[i] if stock[i] < s["rate"] * trail[i] else s["rate"] * trail[i]
            nxt = stock[i] - h + s["regen"]
            stock[i] = s["cap"] if nxt > s["cap"] else nxt
    return alloc


def _local_search(srcs, K, T, A, alloc0, seed, iters):
    """Generic hill-climbing: repeatedly move a random-sized chunk of ants from
    one source to another in one random round; keep the move iff it does not
    decrease total harvest. Deterministic seeded RNG."""
    ni, nf = _rng(seed)
    alloc = [row[:] for row in alloc0]
    best = _simulate(srcs, T, A, alloc)
    for _ in range(iters):
        t = ni(0, T - 1)
        i = ni(0, K - 1)
        j = ni(0, K - 1)
        if i == j or alloc[t][i] <= 0:
            continue
        step = ni(1, alloc[t][i])
        alloc[t][i] -= step
        alloc[t][j] += step
        v = _simulate(srcs, T, A, alloc)
        if v >= best:
            best = v
        else:
            alloc[t][i] += step
            alloc[t][j] -= step
    return alloc, best


def _reference_values(inst):
    """Return (base, ub) for this instance, computed ONCE, independent of any
    candidate. base = equal-split-forever score. ub = 1.08 * best harvest found
    by generic local search seeded from plain (non-phased) starting policies."""
    srcs, K, T, A = inst["sources"], inst["K"], inst["T"], inst["A"]
    base = _simulate(srcs, T, A, _uniform_alloc(K, T, A))
    starts = [_uniform_alloc(K, T, A), _rate_prop_alloc(srcs, K, T, A),
              _myopic_alloc(srcs, K, T, A)]
    best_alloc, best_val = None, -1.0
    for si, s0 in enumerate(starts):
        alloc, val = _local_search(srcs, K, T, A, s0, seed=7000 + si * 13, iters=3500)
        if val > best_val:
            best_val, best_alloc = val, alloc
    for extra in range(2):
        alloc, val = _local_search(srcs, K, T, A, best_alloc, seed=7500 + extra, iters=2500)
        if val > best_val:
            best_val, best_alloc = val, alloc
    ub = best_val * 1.08
    return base, ub


# ----------------------------- validation ----------------------------------
def _validate_alloc(inst, answer):
    if not isinstance(answer, dict):
        return None
    alloc = answer.get("alloc")
    if not isinstance(alloc, list) or len(alloc) != inst["T"]:
        return None
    K, A = inst["K"], inst["A"]
    out = []
    for row in alloc:
        if not isinstance(row, list) or len(row) != K:
            return None
        r = []
        s = 0
        for x in row:
            if isinstance(x, bool) or not isinstance(x, int):
                return None
            if x < 0:
                return None
            r.append(x)
            s += x
        if s > A:
            return None
        out.append(r)
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
        base, ub = _reference_values(inst)
        denom = ub - base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "K": inst["K"], "T": inst["T"], "A": inst["A"],
                  "sources": [dict(s) for s in inst["sources"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            alloc = _validate_alloc(inst, ans)
        except Exception:
            alloc = None
        if alloc is None:
            vec.append(0.0)
            continue
        try:
            harvest = _simulate(inst["sources"], inst["T"], inst["A"], alloc)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (harvest - base) / denom
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
