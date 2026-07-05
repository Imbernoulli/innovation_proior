#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0122 -- "Ridgeline Turnpike: Bounded-Space Gantry Staffing"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  A mountain turnpike clears traffic through a bank of electronic toll
gantries.  Vehicles arrive already grouped into indivisible PLATOONS (convoys that
travel bumper-to-bumper and must be waved through the SAME gantry in one green
cycle).  Each staffed gantry can process a total load of C per cycle (sum of the
platoon "loads" routed to it, e.g. combined axle count).  A platoon is routed
whole -- it may never be split across two gantries.

The catch: powering + staffing a gantry is expensive, and the control cabin can
keep only K gantries OPEN at the same moment.  A gantry is "open" from the cycle it
first accepts a platoon until the cycle it accepts its last platoon; a closed
gantry is dark and cannot be reopened.  At no instant may more than K gantries be
open simultaneously.  The turnpike wants to clear the whole arrival stream using as
FEW gantries as possible.

This is BOUNDED-SPACE online 1-D bin packing skinned as a turnpike:
    platoons  = items (integer loads),
    C         = bin (gantry) capacity,
    K         = max simultaneously-open bins,
    gantries opened = bins used  -> MINIMIZE.
The bounded-space constraint (<=K open bins at once) is what makes this an ONLINE
problem: a candidate cannot lay out an unconstrained offline packing, because
scattering one gantry's platoons across the arrival stream forces too many gantries
to be open at once.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int), "n": N (int), "max_open": K (int),
             "platoons": [s_0, ..., s_{N-1}]   # integer loads, 1 <= s_i <= C,
                                                # in ARRIVAL order (index = cycle)}
  stdout: ONE JSON object:
            {"assign": [g_0, ..., g_{N-1}]}
          where g_i >= 0 is the gantry index platoon i is routed to.  Gantry indices
          need not be contiguous; a gantry "exists" iff >=1 platoon uses it, and the
          number of DISTINCT gantries is the count we minimize.

  A routing is VALID iff ALL hold:
    * `assign` is a list of exactly N non-negative integers,
    * no gantry's total routed load exceeds C,
    * BOUNDED SPACE: for gantry g let first(g)=min arrival index routed to it and
      last(g)=max such index; g is "open" over cycles [first(g), last(g)].  For
      every cycle i in 0..N-1, the number of gantries g with first(g)<=i<=last(g)
      must be <= K.
  Invalid output, wrong length, an overfilled gantry, a bounded-space violation, a
  crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    q_lb   = L1 lower bound = ceil(sum(platoons) / C)            # unreachable ideal
    q_base = gantries used by the internal NEXT-FIT operator      # weak K=1 baseline
    q_cand = gantries used by the candidate routing
  and normalize with an affine anchor (weak baseline -> 0.1, L1 ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate matching next-fit scores ~0.1; one reaching the (generally
  unreachable) L1 bound scores 1.0; doing worse than next-fit scores < 0.1.

  Because L1 is a LOOSE lower bound AND only K gantries may be open at once, even
  strong bounded-space packers stay strictly below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(L1, next-fit baseline) are computed by THIS parent process, so a frame-walking /
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

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _build_platoons(seed, n, C, dist):
    """Return a list of N integer platoon loads in [1, C]. Deterministic."""
    ni = _rng(seed)
    out = []
    for _ in range(n):
        if dist == "cars":                              # mostly small single cars/groups
            s = ni(1, max(1, C // 3))
        elif dist == "mixed":                           # uniform loads
            s = ni(1, C)
        elif dist == "trucks":                          # large platoons, hard to pair
            s = ni(max(1, (2 * C) // 5), (17 * C) // 20)
        elif dist == "rush":                            # many small + some large (bimodal)
            s = ni(1, max(1, C // 5)) if ni(0, 99) < 55 else ni((3 * C) // 5, (9 * C) // 10)
        else:
            s = ni(1, C)
        if s < 1:
            s = 1
        if s > C:
            s = C
        out.append(s)
    return out


def _build_instances():
    """Deterministic instance family. (seed, n, C, K, dist)."""
    specs = [
        (4101, 24, 20, 3, "mixed"),
        (4102, 28, 20, 3, "mixed"),
        (4103, 30, 24, 4, "rush"),
        (4104, 26, 20, 3, "rush"),
        (4105, 32, 24, 4, "mixed"),
        (4106, 30, 18, 3, "cars"),
        (4107, 28, 20, 4, "mixed"),
        (4108, 30, 24, 3, "trucks"),
        # harder / larger held-out instances
        (4211, 44, 22, 3, "mixed"),
        (4212, 42, 20, 4, "rush"),
        (4213, 48, 24, 4, "mixed"),
        (4214, 52, 20, 3, "trucks"),
    ]
    out = []
    for seed, n, C, K, dist in specs:
        platoons = _build_platoons(seed, n, C, dist)
        out.append({"name": f"turnpike{seed}", "capacity": C, "n": n,
                    "max_open": K, "platoons": platoons, "dist": dist})
    return out


# ----------------------------- references ----------------------------------
def _l1(platoons, C):
    return -(-sum(platoons) // C)          # ceil(sum / C)


def _next_fit(platoons, C):
    """Weak online operator: keep filling the current gantry; open a fresh one the
    moment a platoon doesn't fit.  Uses exactly one open gantry at a time (K=1)."""
    bins = 1
    rem = C
    for s in platoons:
        if s <= rem:
            rem -= s
        else:
            bins += 1
            rem = C - s
    return bins


# ----------------------------- validation ----------------------------------
def _gantries(inst, answer):
    """Validate answer (capacity + bounded space) against the instance.
    Return distinct-gantry count on success, else None."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    platoons = inst["platoons"]
    C = inst["capacity"]
    N = inst["n"]
    K = inst["max_open"]
    if len(assign) != N:
        return None
    load = {}
    first = {}
    last = {}
    for i, g in enumerate(assign):
        if isinstance(g, bool) or not isinstance(g, int):
            return None
        if g < 0:
            return None
        load[g] = load.get(g, 0) + platoons[i]
        if load[g] > C:
            return None
        if g not in first:
            first[g] = i
        last[g] = i
    # bounded-space check: at each cycle, count gantries whose [first,last] spans it.
    # Use a sweep of open/close events.
    delta = [0] * (N + 1)
    for g in first:
        delta[first[g]] += 1
        delta[last[g] + 1] -= 1
    cur = 0
    for i in range(N):
        cur += delta[i]
        if cur > K:
            return None
    return len(load)


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
        platoons = inst["platoons"]
        q_lb = _l1(platoons, C)
        q_base = _next_fit(platoons, C)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": C, "n": inst["n"],
                  "max_open": inst["max_open"], "platoons": list(platoons)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _gantries(inst, ans)
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
