#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0340 -- "Server Hall Cooling: Thermal-Aware Job Placement"
(family: heuristic-contest-offline; format B, quality-metric; theme: data-center cooling).

THEME.  A data hall is an N x N grid of rack slots.  Each slot (r, c) has an integer
COOLING capacity `cool[r][c] >= 0` -- how much dissipated heat that slot can absorb
(cold-aisle airflow is strong near the CRAC vents and weak in the dead zones between
them).  You must place J compute JOBS, job j having an integer heat LOAD `loads[j]`,
onto J DISTINCT rack slots.

Heat recirculates locally.  A job of load w placed at slot (r, c) deposits heat onto
its 3x3 neighborhood via a fixed thermal KERNEL:

        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]

i.e. slot (r+dr, c+dc) (clipped to the hall) receives  w * kernel[dr+1][dc+1].  The
heat arriving at a slot is the SUM of deposits from every job whose footprint reaches
it.  A slot runs hot when arriving heat exceeds its cooling:

        over(a, b) = max(0, deposit(a, b) - cool[a][b])

Hotspots damage hardware super-linearly, so the operator MINIMIZES the total squared
over-temperature:

        penalty(placement) = sum over all slots of over(a, b) ** 2

The tension is coverage-vs-clustering, twice over: piling jobs together stacks their
3x3 footprints (a shared slot sees 4*w1 + 2*w2 + ...), and ignoring the cooling map
wastes the strong-vent slots.  A good operator spreads the hottest jobs apart AND onto
high-cooling slots.  This is a quadratic-assignment / facility-dispersion instance
skinned as an offline AtCoder-heuristic-style contest, scored by a fixed deterministic
formula under an operation (not wall-time) budget.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name":  str,
             "n":     N (int),                 # hall is N x N
             "j":     J (int),                 # number of jobs
             "loads": [w0, w1, ..., w_{J-1}],  # int heat load per job, in listed order
             "cool":  [[...], ...],            # N rows of N ints >= 0 (cooling capacity)
             "kernel":[[1,2,1],[2,4,2],[1,2,1]]}  # fixed 3x3 thermal footprint weights
  stdout: ONE JSON object:
            {"place": [[r0, c0], [r1, c1], ...]}   # EXACTLY J entries
          place[j] = [r, c] is the slot for job j (loads[j]).  Every [r, c] is a slot
          with 0 <= r, c < N.  The J slots must be pairwise DISTINCT.

  A placement is VALID iff `place` is a list of exactly J pairs of integers, every
  coordinate is in range, and all J slots are distinct.  Wrong length, out-of-range or
  duplicate slots, non-integer coordinates, a crash, a timeout, or non-JSON -> that
  instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    P_base = penalty of the overlap-BLIND, cooling-BLIND reference: assign job j to the
             row-major slot (j // N, j % N).  Jobs pack into a corner, their footprints
             stack, and strong-vent slots go unused.  This is the weak baseline.
    P_cand = penalty of the candidate's placement.
  normalized with an affine anchor (weak baseline -> 0.1, the unreachable perfect hall
  penalty 0 -> 1.0):
    r = clamp( 0.1 + 0.9 * (P_base - P_cand) / max(1e-9, P_base), 0, 1 )
  Reproducing the row-major reference scores ~0.1; doing WORSE scores < 0.1; cutting the
  squared over-temperature scores higher.  Because total deposited heat exceeds total
  cooling on every instance, penalty 0 is unreachable, so even excellent operators stay
  well below 1.0 -> headroom.  Your final score is the mean of r over all instances (a
  mix of hall sizes, job counts, cooling maps, and harder held-out halls with tighter
  cooling and hotter jobs).

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The reference penalty
and all validation happen in THIS parent process, so a frame-walking / introspecting
candidate learns nothing that helps it place jobs.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# fixed thermal footprint kernel (3x3, offsets -1..+1)
KERNEL = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _build_cool(seed, n, num_vents, base_lo, base_hi, vent_lo, vent_hi, sf):
    """Deterministic N x N cooling map: a few CRAC vents on a weak floor.

    Cooling is deliberately MODEST relative to job deposits (a job's center deposit is
    4*load): total hall cooling is well below total deposited heat, so some over-
    temperature is unavoidable no matter how jobs are placed -> penalty 0 is never
    reachable and even the best operator keeps headroom.
    """
    ni = _rng(seed)
    vents = []
    for _ in range(num_vents):
        vr = ni(0, n - 1); vc = ni(0, n - 1)
        amp = ni(vent_lo, vent_hi)
        vents.append((vr, vc, amp))
    cool = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            v = ni(base_lo, base_hi)
            for (vr, vc, amp) in vents:
                d2 = (r - vr) * (r - vr) + (c - vc) * (c - vc)
                denom = 1 + d2 // sf            # gentle rational falloff
                v += amp // denom
            cool[r][c] = v
    return cool


def _build_loads(seed, j, lo, hi):
    ni = _rng(seed ^ 0x5bd1e995)
    return [ni(lo, hi) for _ in range(j)]


def _build_instances():
    """Deterministic family: (seed, n, j, load_lo, load_hi, num_vents,
       base_lo, base_hi, vent_lo, vent_hi, spread_factor)."""
    specs = [
        (301, 6, 14, 10, 34, 3, 5, 14, 24, 55, 2),
        (302, 6, 16, 10, 34, 2, 5, 12, 24, 55, 2),
        (303, 7, 18, 10, 36, 3, 5, 14, 24, 60, 2),
        (304, 7, 20, 10, 36, 3, 4, 12, 24, 58, 2),
        (305, 8, 24, 10, 38, 4, 5, 14, 26, 62, 2),
        (306, 8, 26, 10, 38, 3, 4, 12, 26, 60, 2),
        (307, 9, 30, 10, 40, 4, 5, 14, 26, 64, 3),
        (308, 9, 34, 10, 40, 4, 4, 12, 26, 62, 3),
        # harder / held-out: tighter cooling floor, hotter jobs, more crowding
        (411, 8, 30, 14, 48, 3, 3, 9, 22, 55, 2),
        (412, 9, 40, 14, 50, 4, 3, 9, 22, 58, 2),
        (413, 9, 44, 16, 52, 3, 3, 8, 22, 55, 2),
        (414, 10, 52, 16, 54, 4, 3, 9, 22, 58, 2),
    ]
    out = []
    for (seed, n, j, load_lo, load_hi, nv, bl, bh, vl, vh, sf) in specs:
        cool = _build_cool(seed, n, nv, bl, bh, vl, vh, sf)
        loads = _build_loads(seed, j, load_lo, load_hi)
        out.append({"name": f"hall{seed}", "n": n, "j": j,
                    "loads": loads, "cool": cool})
    return out


# ----------------------------- references / scoring ------------------------
def _penalty(n, cool, loads, place):
    """place: list of (r,c) per job; deposit via KERNEL then sum of squared overheat."""
    dep = [[0] * n for _ in range(n)]
    for j in range(len(place)):
        r, c = place[j]
        w = loads[j]
        for dr in (-1, 0, 1):
            rr = r + dr
            if rr < 0 or rr >= n:
                continue
            krow = KERNEL[dr + 1]
            drow = dep[rr]
            for dc in (-1, 0, 1):
                cc = c + dc
                if cc < 0 or cc >= n:
                    continue
                drow[cc] += w * krow[dc + 1]
    tot = 0
    for r in range(n):
        cr = cool[r]; dr_ = dep[r]
        for c in range(n):
            ov = dr_[c] - cr[c]
            if ov > 0:
                tot += ov * ov
    return tot


def _baseline_place(n, j):
    return [(idx // n, idx % n) for idx in range(j)]


def _validate(inst, answer):
    """Validate placement against the instance. Return list of J (r,c) or None."""
    if not isinstance(answer, dict):
        return None
    place = answer.get("place")
    if not isinstance(place, list):
        return None
    n = inst["n"]; j = inst["j"]
    if len(place) != j:
        return None
    seen = set()
    out = []
    for pair in place:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        r, c = pair[0], pair[1]
        if isinstance(r, bool) or isinstance(c, bool):
            return None
        if not isinstance(r, int) or not isinstance(c, int):
            return None
        if r < 0 or r >= n or c < 0 or c >= n:
            return None
        if (r, c) in seen:
            return None
        seen.add((r, c))
        out.append((r, c))
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        n = inst["n"]; j = inst["j"]; loads = inst["loads"]; cool = inst["cool"]
        p_base = _penalty(n, cool, loads, _baseline_place(n, j))
        denom = p_base if p_base > 1e-9 else 1e-9
        public = {"name": inst["name"], "n": n, "j": j,
                  "loads": list(loads), "cool": [list(row) for row in cool],
                  "kernel": [list(row) for row in KERNEL]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            place = _validate(inst, ans)
        except Exception:
            place = None
        if place is None:
            vec.append(0.0)
            continue
        p_cand = _penalty(n, cool, loads, place)
        r = 0.1 + 0.9 * (p_base - p_cand) / denom
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
