#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0161 -- "Green-Wave Phasing: Coordinating a Traffic-Signal
Grid" (family: seeded-numerical-sim; format B, quality-metric; inspired by Frontier-Eng
Optics/Communications, skinned as a traffic-signal grid).

THEME.  An authority controls an N x N grid of traffic signals that all share one fixed
cycle time; the only free variable is each signal's OFFSET (green phase) theta[r][c], a
real angle in radians.  A fixed, seeded wave simulator treats each intersection as a
unit oscillator with complex amplitude exp(i*theta) and computes the grid's FLOW SPECTRUM
as the centered 2-D DFT of that field; the flow INTENSITY at a spectrum point is the
squared magnitude there.  Total intensity over the whole spectrum is a constant (= n^4)
independent of the offsets -- offsets only REDISTRIBUTE flow.

The city designates M TARGET CORRIDORS: M off-center points of the flow spectrum that
should carry high, balanced throughput (a "spot array").  The designer chooses offsets to
maximize a composite of diffraction EFFICIENCY (fraction of total flow landing on the
corridors) and UNIFORMITY (evenness of the corridor intensities).  This is a phase-mask /
spot-array-generator problem: a phase-only field can pour energy onto the corridors but
cannot make them all perfectly bright AND perfectly equal, so the achievable score has
permanent headroom below the ideal.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n": N (int),                 # grid & spectrum are N x N
             "spots": [[u,v], ...]}        # M target corridors, 0<=u,v<N, DC excluded
  stdout: ONE JSON object:
            {"phases": [[theta,...], ...]} # EXACTLY N rows of N real finite floats

  An answer is VALID iff `phases` is an N x N array of real finite numbers.  Wrong shape,
  a non-numeric / NaN / Inf entry, a crash, a timeout, or non-JSON -> that instance
  scores 0.0.

SCORING (deterministic; no wall-time).  The evaluator SIMULATES the flow spectrum itself:
    U          = exp(i*phases);  F = fftshift(fft2(U));  I = |F|^2
    E_total    = sum(I)                                     # = n^4, constant
    efficiency = (sum of I over the M spots) / E_total      # in [0,1]
    uniformity = (min spot I) / (max spot I)  (0 if all spots dark)
    M_cand     = efficiency * uniformity                    # composite in [0,1]
  References (in THIS parent only):
    M_base = composite of the all-zero-offset design (all flow at DC) = 0  (weak baseline)
    M_ub   = 1.0                                            # loose unreachable upper bound
  Affine anchor (weak baseline -> 0.1, ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (M_cand - M_base) / (M_ub - M_base), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  All simulation, all
references, and all validation happen in THIS parent process, so a frame-walking /
introspecting candidate learns nothing that helps it.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import numpy as np
import isorun


# ----------------------------- instance family -----------------------------
def _make_spots(seed, n, m):
    """Deterministic set of M distinct off-center spectrum points (DC excluded)."""
    r = np.random.default_rng(seed)
    c = n // 2
    forbidden = {(c, c)}
    pts = []
    while len(pts) < m:
        u = int(r.integers(1, n - 1))
        v = int(r.integers(1, n - 1))
        if (u, v) in forbidden:
            continue
        forbidden.add((u, v))
        pts.append([u, v])
    return pts


def _build_instances():
    """Deterministic instance family: (seed, n, m). Later rows = harder held-out grids."""
    specs = [
        (11, 8, 6),
        (12, 10, 6),
        (13, 10, 8),
        (14, 12, 8),
        (15, 12, 10),
        (16, 14, 9),
        (17, 14, 12),
        (18, 16, 10),
        # harder / larger held-out grids (bigger n, more corridors)
        (19, 16, 14),
        (20, 18, 12),
        (21, 20, 14),
        (22, 20, 16),
    ]
    out = []
    for (seed, n, m) in specs:
        out.append({"name": f"grid{seed}", "n": n,
                    "spots": _make_spots(seed, n, m)})
    return out


# ----------------------------- simulation / metrics ------------------------
def _composite(theta, spots):
    """Simulate the flow spectrum for offset array `theta` and return the composite M.

    M = efficiency * uniformity -- BOTH must be high (a corridor left dark drives the
    whole score to 0), which is the standard holographic spot-array quality metric.
    """
    U = np.exp(1j * theta)
    F = np.fft.fftshift(np.fft.fft2(U))
    I = np.abs(F) ** 2
    E_total = float(I.sum())
    if E_total <= 1e-12:
        return 0.0
    sv = np.array([I[u, v] for (u, v) in spots], dtype=float)
    eff = float(sv.sum()) / E_total
    mx = float(sv.max())
    unif = (float(sv.min()) / mx) if mx > 1e-12 else 0.0
    M = eff * unif
    if not np.isfinite(M):
        return 0.0
    return max(0.0, min(1.0, M))


def _baseline_composite(inst):
    """All-zero offsets: flow collapses to DC, every corridor dark -> M_base = 0."""
    n = inst["n"]
    theta = np.zeros((n, n), dtype=float)
    return _composite(theta, inst["spots"])


def _validate(inst, answer):
    """Validate the candidate answer. Return an n x n float ndarray or None."""
    if not isinstance(answer, dict):
        return None
    phases = answer.get("phases")
    if not isinstance(phases, list):
        return None
    n = inst["n"]
    if len(phases) != n:
        return None
    grid = []
    for row in phases:
        if not isinstance(row, list) or len(row) != n:
            return None
        out_row = []
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return None
            xf = float(x)
            if xf != xf or xf in (float("inf"), float("-inf")):
                return None
            out_row.append(xf)
        grid.append(out_row)
    arr = np.array(grid, dtype=float)
    if arr.shape != (n, n) or not np.all(np.isfinite(arr)):
        return None
    return arr


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    M_ub = 1.0
    vec = []
    for inst in instances:
        M_base = _baseline_composite(inst)          # = 0 for the all-zero design
        denom = M_ub - M_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "n": inst["n"],
                  "spots": [list(p) for p in inst["spots"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            theta = _validate(inst, ans)
        except Exception:
            theta = None
        if theta is None:
            vec.append(0.0)
            continue
        M_cand = _composite(theta, inst["spots"])
        r = 0.1 + 0.9 * (M_cand - M_base) / denom
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
