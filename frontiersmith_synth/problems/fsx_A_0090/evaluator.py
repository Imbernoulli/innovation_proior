#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0090 -- "Rift Valley Geothermal: Well-Field Siting"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  A geothermal operator surveys a rectangular tract of the Rift Valley and
produces a discrete heat map: an N x N grid where cell (r, c) holds an integer
`heat` (recoverable thermal energy, arbitrary units).  Heat is concentrated in a
handful of underground plumes (hotspots) with a diffuse background everywhere.

The operator may drill K production wells at K DISTINCT grid cells.  A well drilled
at (r, c) taps every cell within Chebyshev radius R of it -- the (2R+1)x(2R+1)
square window centered on the well, clipped to the tract.  Because the reservoir is
a shared fluid body, once a cell's heat has been drawn by ANY well it cannot be
drawn again: the field's yield is the heat of the UNION of all wells' windows.

    yield(placement) = sum of heat[r][c] over the set of cells covered by >=1 well.

The operator MAXIMIZES yield.  Two wells drilled close together waste capacity by
re-tapping the same cells, so the interesting tension is coverage vs. clustering:
the hottest cells all sit on top of the tallest plume, but piling wells there
double-taps it, while spreading wells across separate plumes captures more total
heat.  This is a max-coverage / facility-dispersion instance skinned as an offline
AtCoder-heuristic-style siting contest, scored by a fixed deterministic formula.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n": N (int),                 # grid is N x N
             "radius": R (int),            # each well taps a Chebyshev-R window
             "k": K (int),                 # number of wells to drill
             "heat": [[...], ...]}         # N rows of N ints >= 0
  stdout: ONE JSON object:
            {"wells": [[r0, c0], [r1, c1], ...]}   # EXACTLY K entries
          Each [r, c] is a grid cell with 0 <= r, c < N.  The K cells must be
          pairwise DISTINCT.

  A placement is VALID iff `wells` is a list of exactly K pairs of integers, every
  coordinate is in range, and all K cells are distinct.  Wrong length, out-of-range
  or duplicate cells, non-integer coordinates, a crash, a timeout, or non-JSON ->
  that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    y_base = union yield of the K HOTTEST cells (top-K by heat, tie-break by (r,c)).
             This overlap-BLIND placement piles wells onto the tallest plume; it is
             the weak baseline.
    y_ub   = total heat of the ENTIRE tract (sum of every cell).  A loose, generally
             unreachable upper bound (no K windows can cover the whole tract plus its
             diffuse background), so even excellent siters stay below 1.0 -> headroom.
    y_cand = union yield of the candidate's placement.
  normalized with an affine anchor (weak baseline -> 0.1, full-tract ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (y_cand - y_base) / max(1e-9, y_ub - y_base), 0, 1 )
  Reproducing the top-K-hottest placement scores ~0.1; doing worse scores < 0.1;
  spreading wells to cover more distinct heat scores higher, capped at 1.0.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  Every reference
(baseline, upper bound) and all validation happen in THIS parent process, so a
frame-walking / introspecting candidate learns nothing that helps it site wells.

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
def _build_heat(seed, n, num_hot):
    """Deterministic N x N integer heat map: `num_hot` plumes on a diffuse floor."""
    ni = _rng(seed)
    # place plumes with distinct amplitudes / spreads
    hots = []
    for _ in range(num_hot):
        hr = ni(0, n - 1)
        hc = ni(0, n - 1)
        amp = ni(40, 100)          # peak strength
        sig = ni(1, 3)             # spread (in cells); small -> tight plume
        hots.append((hr, hc, amp, sig))
    heat = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            v = ni(0, 4)           # diffuse background floor (nonzero everywhere)
            for (hr, hc, amp, sig) in hots:
                d2 = (r - hr) * (r - hr) + (c - hc) * (c - hc)
                # integer approximation of amp * exp(-d2 / (2 sig^2)) via a rational falloff
                denom = 1 + (d2 * 100) // (2 * sig * sig)
                v += (amp * 100) // denom
            heat[r][c] = v
    return heat


def _build_instances():
    """Deterministic instance family: (seed, n, radius, k, num_hot)."""
    specs = [
        (101, 14, 2, 4, 4),
        (102, 16, 2, 5, 5),
        (103, 16, 2, 4, 6),
        (104, 18, 3, 5, 5),
        (105, 18, 2, 6, 6),
        (106, 20, 2, 6, 5),
        (107, 20, 3, 5, 7),
        (108, 22, 3, 6, 6),
        # harder / larger held-out instances (more plumes than wells)
        (211, 22, 2, 6, 9),
        (212, 24, 3, 7, 8),
        (213, 24, 2, 7, 10),
        (214, 26, 3, 8, 9),
    ]
    out = []
    for (seed, n, radius, k, num_hot) in specs:
        heat = _build_heat(seed, n, num_hot)
        out.append({"name": f"tract{seed}", "n": n, "radius": radius,
                    "k": k, "heat": heat})
    return out


# ----------------------------- references / scoring ------------------------
def _window_cells(r, c, R, n):
    r0 = max(0, r - R); r1 = min(n - 1, r + R)
    c0 = max(0, c - R); c1 = min(n - 1, c + R)
    return r0, r1, c0, c1


def _union_yield(heat, wells, R, n):
    covered = set()
    for (r, c) in wells:
        r0, r1, c0, c1 = _window_cells(r, c, R, n)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                covered.add(rr * n + cc)
    tot = 0
    for idx in covered:
        tot += heat[idx // n][idx % n]
    return tot


def _baseline_yield(heat, R, n, k):
    """Top-K hottest cells (tie-break by row then col), overlap-blind."""
    cells = [(heat[r][c], r, c) for r in range(n) for c in range(n)]
    cells.sort(key=lambda t: (-t[0], t[1], t[2]))
    wells = [(r, c) for (_, r, c) in cells[:k]]
    return _union_yield(heat, wells, R, n)


def _validate(inst, answer):
    """Validate placement against the instance. Return list of K (r,c) or None."""
    if not isinstance(answer, dict):
        return None
    wells = answer.get("wells")
    if not isinstance(wells, list):
        return None
    n = inst["n"]; k = inst["k"]
    if len(wells) != k:
        return None
    seen = set()
    out = []
    for pair in wells:
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
        heat = inst["heat"]; n = inst["n"]; R = inst["radius"]; k = inst["k"]
        y_base = _baseline_yield(heat, R, n, k)
        y_ub = sum(sum(row) for row in heat)
        denom = y_ub - y_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "n": n, "radius": R, "k": k,
                  "heat": [list(row) for row in heat]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            wells = _validate(inst, ans)
        except Exception:
            wells = None
        if wells is None:
            vec.append(0.0)
            continue
        y_cand = _union_yield(heat, wells, R, n)
        r = 0.1 + 0.9 * (y_cand - y_base) / denom
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
