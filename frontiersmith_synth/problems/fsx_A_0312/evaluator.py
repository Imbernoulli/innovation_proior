#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0312 -- "Belt Rush: Hauler Stowage for a Mining Shift"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  A robotic rig cracks an asteroid.  Over one mining shift a stream of ore
FRAGMENTS is spat out, each with an integer MASS (arbitrary units).  Every cargo
HAULER can carry up to CAPACITY units of mass.  You must stow EVERY fragment into
some hauler so that no hauler is ever over capacity, and you want to dispatch as FEW
haulers as possible (each dispatched hauler costs fuel).  This is 1-D bin packing
skinned as an asteroid-mining stowage contest.

THE FIXED STREAMING SIMULATOR (the reference).  The rig also runs a dumb on-board
autoloader: it keeps ONE hauler docked, drops each arriving fragment into it, and
the instant a fragment does not fit it seals that hauler, dispatches it, and docks a
fresh one (the classic Next-Fit online rule).  The number of haulers this autoloader
uses is the WEAK BASELINE the candidate must beat.  Because Next-Fit can never
back-fill a sealed hauler, it wastes capacity -- a smarter offline stowage plan does
much better.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "capacity": C (int > 0),
             "masses": [m0, m1, ..., m_{n-1}]}   # n ints, each 1 <= m_i <= C,
                                                 # in ARRIVAL order
  stdout: ONE JSON object:
            {"assign": [b0, b1, ..., b_{n-1}]}   # b_i = index of the hauler that
                                                 # fragment i is stowed in
          Each b_i is an integer with 0 <= b_i < n (you may open at most n haulers;
          indices need not be contiguous -- only the COUNT of distinct haulers used
          is charged).

  A plan is VALID iff `assign` is a list of exactly n integers, every b_i is in
  [0, n), no b_i is a bool, and for every hauler the total mass of the fragments
  assigned to it is <= C.  Wrong length, out-of-range index, an over-capacity hauler,
  a non-integer entry, a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    nf  = haulers used by the fixed Next-Fit streaming simulator (weak baseline).
    lb  = L1 lower bound = ceil(sum(masses) / C)  (no feasible plan can beat this;
          it is generally NOT achievable because of packing waste -> leaves headroom
          so even a strong stower stays below 1.0).
    cnd = distinct haulers used by the candidate's (valid) plan; note cnd >= lb.
  normalized with an affine anchor (reproduce Next-Fit -> 0.1, reach the L1 bound
  -> 1.0):
    r = clamp( 0.1 + 0.9 * (nf - cnd) / max(nf - lb, 1), 0, 1 )
  Reproducing the streaming autoloader scores ~0.1; using more haulers scores < 0.1;
  every hauler you save versus Next-Fit moves you toward 1.0, capped there.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The baseline, the
lower bound, and all validation are computed in THIS parent process, so a
frame-walking / introspecting candidate learns nothing that helps it pack.

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


def _unit(ni):
    # deterministic float in (0, 1]
    return (ni(1, 1000000)) / 1000000.0


# ----------------------------- instance family -----------------------------
def _gen_masses(seed, n, C, dist):
    """Deterministic list of n integer masses in [1, C] under distribution `dist`."""
    ni = _rng(seed)
    out = []
    for _ in range(n):
        if dist == "uni_half":                       # uniform, small-ish fragments
            m = ni(1, max(2, C // 2))
        elif dist == "uni_23":                       # uniform up to two thirds
            m = ni(1, max(2, (2 * C) // 3))
        elif dist == "skew":                         # bimodal: many small, some big
            if ni(1, 100) <= 60:
                m = ni(1, max(2, C // 5))
            else:
                m = ni(max(2, C // 3), max(3, (2 * C) // 3))
        elif dist == "weibull":                      # skewed toward small (u^2 falloff)
            u = _unit(ni)
            m = 1 + int((C - 1) * (u * u))
        elif dist == "uni_full":                     # full-range uniform (forced waste)
            m = ni(1, C)
        elif dist == "biglarge":                     # mostly large fragments (heavy waste)
            if ni(1, 100) <= 40:
                m = ni(1, C)
            else:
                m = ni(C // 2, C)
        else:                                        # fallback uniform
            m = ni(1, C)
        if m < 1:
            m = 1
        if m > C:
            m = C
        out.append(m)
    return out


def _build_instances():
    """Deterministic instance family: (seed, n, capacity, dist)."""
    specs = [
        (101, 30, 100, "uni_full"),
        (102, 40, 120, "uni_full"),
        (103, 40, 150, "uni_23"),
        (104, 50, 200, "biglarge"),
        (105, 50, 100, "uni_full"),
        (106, 60, 250, "uni_23"),
        (107, 60, 300, "uni_full"),
        (108, 45, 180, "biglarge"),
        # harder / larger held-out instances
        (211, 70, 200, "uni_full"),
        (212, 70, 240, "biglarge"),
        (213, 80, 100, "uni_full"),
        (214, 80, 320, "uni_23"),
    ]
    out = []
    for (seed, n, C, dist) in specs:
        masses = _gen_masses(seed, n, C, dist)
        out.append({"name": f"shift{seed}", "capacity": C, "masses": masses})
    return out


# ----------------------------- references / scoring ------------------------
def _next_fit(masses, C):
    """Classic Next-Fit online rule -> number of haulers used (the weak baseline)."""
    used = 0
    load = None
    for m in masses:
        if load is None or load + m > C:
            used += 1
            load = m
        else:
            load += m
    return used


def _l1_bound(masses, C):
    return (sum(masses) + C - 1) // C


def _validate(inst, answer):
    """Validate a stowage plan. Return the number of distinct haulers used, or None."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    masses = inst["masses"]
    C = inst["capacity"]
    n = len(masses)
    if len(assign) != n:
        return None
    loads = {}
    for i, b in enumerate(assign):
        if isinstance(b, bool) or not isinstance(b, int):
            return None
        if b < 0 or b >= n:
            return None
        loads[b] = loads.get(b, 0) + masses[i]
        if loads[b] > C:
            return None
    return len(loads)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        masses = inst["masses"]
        C = inst["capacity"]
        nf = _next_fit(masses, C)
        lb = _l1_bound(masses, C)
        denom = max(nf - lb, 1)
        public = {"name": inst["name"], "capacity": C,
                  "masses": list(masses)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            cnd = _validate(inst, ans)
        except Exception:
            cnd = None
        if cnd is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (nf - cnd) / denom
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
