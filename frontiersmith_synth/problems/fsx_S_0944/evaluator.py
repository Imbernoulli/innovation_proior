#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0944 -- "Depot Loading Under a Silent Surge"
(family: opportunity-cost-online-binpack; format B, quality-metric).

THEME.  A transfer depot receives a stream of parcels of unknown size mix.  Each
parcel must be loaded into a truck THE MOMENT it arrives -- but the dispatcher has
a small, bounded RE-LOADING budget: a fixed number of parcels may later be moved
from one already-loaded truck to another (e.g. during a lull) to make room.  Some
days the parcel stream quietly shifts from small/medium parcels to a late SURGE of
oversized parcels; a dispatcher who packs every truck as tight as possible early on
leaves no truck able to absorb the surge, and by the time the surge is recognised the
re-loading budget is too small to fix more than a few trucks.  A dispatcher who reads
the early stream, reserves headroom, and spends the re-load budget *pre-emptively* to
consolidate near-full trucks right as the surge starts does far better.

MECHANISM COMPOSITION (all three shape the objective; no single one solves it):
  (1) arrival-size-distribution-probe: the true arrival mix is not told to you --
      only inferable from the sizes actually seen so far (a prefix statistic).
  (2) reserve-headroom-bestfit: initial placement is online (irrevocable at time of
      arrival) and must trade tight best-fit packing against reserving slack for a
      possible large-item mode.
  (3) bounded-repack-consolidation: a strictly bounded number of post-hoc moves lets
      you patch mistakes / consolidate, but the budget is far too small to just
      re-solve the whole instance -- timing and selection of merges matters.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int), "n": N (int),
             "sizes": [s_0, ..., s_{N-1}],       # arrival order, 1 <= s_i <= C
             "repack_budget": B (int)}
  stdout: ONE JSON object:
            {"placements": [b_0, ..., b_{N-1}],  # truck id parcel i is loaded into
                                                  # AT ARRIVAL (0 <= b_i <= N)
             "moves": [{"after": t, "item": i, "to": b}, ...]}   # optional, <= B
          A move {"after": t, "item": i, "to": b} relocates already-arrived parcel i
          (i <= t) to truck b once parcel t has been loaded; "after" values across
          the list must be non-decreasing (moves replay in the order given).

  VALIDITY: `placements` has length N of ints in [0,N]; no truck's load ever exceeds
  C at any point in the simulated timeline (initial load OR after any move); every
  move's item has already arrived; len(moves) <= B.  Any violation, wrong shape,
  crash, timeout, or non-JSON output -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes, itself:
    q_lb   = L1 lower bound = ceil(sum(sizes) / C)                  # unreachable ideal
    q_base = trucks used by an internal NEXT-FIT-COMMIT operator     # weak online ref
    q_cand = trucks with load > 0 at the END of the candidate's simulated timeline
  and normalizes with an affine anchor (weak baseline -> 0.1, L1 ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  Matching next-fit-commit exactly scores ~0.1; reaching the (generally unreachable)
  L1 bound scores 1.0; doing worse than next-fit-commit scores below 0.1.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (no `dist` label, no
hidden stats).  All references (L1, next-fit-commit) and the full simulation/replay
of `placements`+`moves` are computed by THIS parent process.

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


def _clamp(s, C):
    if s < 1:
        s = 1
    if s > C:
        s = C
    return s


# ----------------------------- instance family ------------------------------
def _build_sizes(seed, n, C, dist):
    ni = _rng(seed)
    out = []
    if dist == "uniform":
        for _ in range(n):
            out.append(ni(1, C))
    elif dist == "medium_random":
        lo, hi = max(1, int(0.35 * C)), max(2, int(0.65 * C))
        for _ in range(n):
            out.append(ni(lo, hi))
    elif dist == "bimodal_mixed":
        slo, shi = 1, max(1, int(0.22 * C))
        llo, lhi = max(1, int(0.60 * C)), max(1, int(0.90 * C))
        for _ in range(n):
            out.append(ni(slo, shi) if ni(0, 99) < 55 else ni(llo, lhi))
    elif dist == "heavy_tail":
        slo, shi = 1, max(1, int(0.20 * C))
        llo, lhi = max(1, int(0.55 * C)), max(1, int(0.90 * C))
        for _ in range(n):
            out.append(ni(slo, shi) if ni(0, 99) < 20 else ni(llo, lhi))
    elif dist in ("burst_trap", "burst_trap_severe", "burst_trap_holdout"):
        if dist == "burst_trap_severe":
            k = int(0.80 * n)
            mlo, mhi = max(1, int(0.38 * C)), max(2, int(0.62 * C))
            llo, lhi = max(1, int(0.72 * C)), max(1, int(0.97 * C))
        else:
            k = int(0.70 * n)
            mlo, mhi = max(1, int(0.40 * C)), max(2, int(0.60 * C))
            llo, lhi = max(1, int(0.65 * C)), max(1, int(0.92 * C))
        for _ in range(k):
            out.append(ni(mlo, mhi))
        for _ in range(n - k):
            out.append(ni(llo, lhi))
    elif dist == "double_burst_trap":
        seg = n // 4
        mlo, mhi = max(1, int(0.40 * C)), max(2, int(0.58 * C))
        llo, lhi = max(1, int(0.68 * C)), max(1, int(0.92 * C))
        for _ in range(seg):
            out.append(ni(mlo, mhi))
        for _ in range(seg):
            out.append(ni(llo, lhi))
        for _ in range(seg):
            out.append(ni(mlo, mhi))
        for _ in range(n - 3 * seg):
            out.append(ni(llo, lhi))
    elif dist == "ramp_trap":
        for i in range(n):
            frac = i / max(1, n - 1)
            lo = int((0.22 + 0.55 * frac) * C)
            hi = int((0.34 + 0.60 * frac) * C)
            hi = max(hi, lo + 1)
            out.append(ni(max(1, lo), max(2, hi)))
    elif dist == "early_large_then_medium":
        k = int(0.30 * n)
        llo, lhi = max(1, int(0.65 * C)), max(1, int(0.92 * C))
        mlo, mhi = max(1, int(0.35 * C)), max(2, int(0.55 * C))
        for _ in range(k):
            out.append(ni(llo, lhi))
        for _ in range(n - k):
            out.append(ni(mlo, mhi))
    else:
        for _ in range(n):
            out.append(ni(1, C))
    return [_clamp(s, C) for s in out]


def _build_instances():
    specs = [
        (301, 30, 30, "uniform"),
        (302, 34, 28, "medium_random"),
        (303, 36, 32, "bimodal_mixed"),
        (304, 32, 26, "heavy_tail"),
        (305, 40, 30, "early_large_then_medium"),
        (306, 40, 30, "burst_trap"),
        (307, 44, 28, "burst_trap_severe"),
        (308, 48, 32, "double_burst_trap"),
        (309, 42, 30, "ramp_trap"),
        (310, 64, 34, "burst_trap_holdout"),
    ]
    out = []
    for seed, n, C, dist in specs:
        sizes = _build_sizes(seed, n, C, dist)
        budget = max(4, round(0.11 * n))
        out.append({"name": f"depot{seed}", "capacity": C, "n": n,
                    "sizes": sizes, "repack_budget": budget, "dist": dist})
    return out


# ----------------------------- references ------------------------------------
def _l1(sizes, C):
    return -(-sum(sizes) // C)  # ceil(sum / C)


def _next_fit_commit(sizes, C):
    """Weak online reference: keep filling the current truck; the moment a
    parcel doesn't fit, dispatch a new one. Never revisits, never repacks."""
    bins = 1
    rem = C
    for s in sizes:
        if s <= rem:
            rem -= s
        else:
            bins += 1
            rem = C - s
    return bins


# ----------------------------- validation / simulation ------------------------
def _final_trucks(inst, answer):
    """Replay placements + moves against the instance. Return final truck count
    or None if the answer is malformed / infeasible / over budget."""
    if not isinstance(answer, dict):
        return None
    placements = answer.get("placements")
    moves = answer.get("moves", [])
    if moves is None:
        moves = []
    n = inst["n"]
    C = inst["capacity"]
    sizes = inst["sizes"]
    budget = inst["repack_budget"]

    if not isinstance(placements, list) or len(placements) != n:
        return None
    if not isinstance(moves, list):
        return None
    if len(moves) > budget:
        return None

    for b in placements:
        if isinstance(b, bool) or not isinstance(b, int):
            return None
        if b < 0 or b > n:
            return None

    parsed = []
    prev_after = -1
    for mv in moves:
        if not isinstance(mv, dict):
            return None
        a = mv.get("after")
        it = mv.get("item")
        to = mv.get("to")
        for v in (a, it, to):
            if isinstance(v, bool) or not isinstance(v, int):
                return None
        if a < 0 or a >= n:
            return None
        if it < 0 or it > a:
            return None
        if to < 0 or to > n:
            return None
        if a < prev_after:
            return None
        prev_after = a
        parsed.append((a, it, to))

    load = [0] * (n + 1)
    item_bin = [-1] * n
    mi = 0
    for t in range(n):
        b = placements[t]
        s = sizes[t]
        load[b] += s
        if load[b] > C:
            return None
        item_bin[t] = b
        while mi < len(parsed) and parsed[mi][0] == t:
            a, it, to = parsed[mi]
            mi += 1
            cur = item_bin[it]
            if cur < 0:
                return None
            s_it = sizes[it]
            if to == cur:
                continue
            if load[to] + s_it > C:
                return None
            load[cur] -= s_it
            load[to] += s_it
            item_bin[it] = to
    if mi != len(parsed):
        return None
    return sum(1 for x in load if x > 0)


# ----------------------------- scoring driver ----------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        C = inst["capacity"]
        sizes = inst["sizes"]
        q_lb = _l1(sizes, C)
        q_base = _next_fit_commit(sizes, C)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": C, "n": inst["n"],
                  "sizes": list(sizes), "repack_budget": inst["repack_budget"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _final_trucks(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None or q_cand <= 0:
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
