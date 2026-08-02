#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1167 (gravity-density-inversion, format C).

The submission IS a density grid rho[z][c] (z=1..nz shallow->deep, c=0..nx-1). It is scored
against a HIDDEN true body (one contiguous column span at one depth layer) that is regenerated
here from the testId printed in <in> -- byte-identical logic to gen.py, never shipped as a
readable ground-truth file. Score = 0.6 * IoU(recovered-active-cells, true-cells)
                                    + 0.4 * (1 - mean capped relative field-misfit),
misfit measured at BOTH the printed (given) stations AND the held-out (odd-column) stations
that were never shown to the solver, normalized against the checker's own trivial reference
construction (a single-cell explain-the-peak guess -- exactly solutions/trivial.py).
"""
import sys, math, random

CELL_W = 2.0
LAYER_H = 2.0
KGAIN = 100.0
RHO_MAX_CONST = 10.0
NOISE_HALF = 0.02
W_IOU = 0.6
W_FIT = 0.4
CAP = 0.92
MAX_TOKENS = 20000

TABLE = {
    1:  (8,  4, 1, 2, 0.60),
    2:  (8,  4, 4, 2, 0.80),
    3:  (10, 5, 2, 3, 0.65),
    4:  (10, 5, 5, 3, 0.85),
    5:  (12, 5, 1, 3, 0.55),
    6:  (12, 6, 5, 3, 0.80),
    7:  (14, 6, 6, 4, 0.85),
    8:  (14, 6, 2, 4, 0.60),
    9:  (16, 7, 6, 4, 0.80),
    10: (16, 7, 7, 3, 0.90),
}


def out_ratio(v, reason=""):
    if reason:
        sys.stdout.write("# %s\n" % reason)
    sys.stdout.write("Ratio: %.6f\n" % v)
    sys.exit(0)


def depth(z):
    return (z - 0.5) * LAYER_H


def kernel(dc, z):
    dx = dc * CELL_W
    dpt = depth(z)
    return KGAIN * dpt / (dx * dx + dpt * dpt) ** 1.5


def true_body(test_id):
    if test_id not in TABLE:
        return None
    nx, nz, z_true, w_true, rho_frac = TABLE[test_id]
    rho_true = rho_frac * RHO_MAX_CONST
    rng = random.Random(9001 + test_id * 131)
    c_lo = rng.randint(0, nx - w_true)
    c_hi = c_lo + w_true - 1
    true_mass = rho_true * w_true
    mass_max = round(1.8 * true_mass, 3)
    return nx, nz, z_true, c_lo, c_hi, rho_true, mass_max


def true_field(cs, z_true, c_lo, c_hi, rho_true):
    return rho_true * sum(kernel(cs - c, z_true) for c in range(c_lo, c_hi + 1))


def noise_factor(test_id, c):
    r = random.Random(5000 + test_id * 97 + c * 13)
    return 1.0 + r.uniform(-NOISE_HALF, NOISE_HALF)


def read_instance(path):
    toks = open(path).read().split()
    idx = 0
    test_id = int(toks[idx]); idx += 1
    nx = int(toks[idx]); idx += 1
    nz = int(toks[idx]); idx += 1
    rho_max = float(toks[idx]); idx += 1
    mass_max = float(toks[idx]); idx += 1
    ns_given = int(toks[idx]); idx += 1
    given = []
    for _ in range(ns_given):
        c = int(toks[idx]); idx += 1
        r = float(toks[idx]); idx += 1
        given.append((c, r))
    return test_id, nx, nz, rho_max, mass_max, given


def predict(grid, nx, nz, cs):
    s = 0.0
    for z in range(1, nz + 1):
        row = grid[z]
        for c in range(nx):
            v = row[c]
            if v != 0.0:
                s += v * kernel(cs - c, z)
    return s


def trivial_grid(nx, nz, rho_max, mass_max, given):
    """The checker's own baseline: explain only the single peak given reading with one
    shallowest-layer cell (identical construction to solutions/trivial.py)."""
    grid = [[0.0] * nx for _ in range(nz + 1)]
    c_peak, r_peak = max(given, key=lambda x: x[1])
    mag = r_peak / kernel(0, 1)
    if mag < 0.0:
        mag = 0.0
    if mag > rho_max:
        mag = rho_max
    grid[1][c_peak] = mag
    total = mag
    if total > mass_max and total > 0:
        f = mass_max / total
        grid[1][c_peak] *= f
    return grid


def score_grid(grid, nx, nz, rho_max, mass_max, z_true, c_lo, c_hi, rho_true,
               given, held):
    total = 0.0
    for z in range(1, nz + 1):
        for c in range(nx):
            total += grid[z][c]
    if total > mass_max * (1.0 + 1e-6):
        return None  # caller treats as infeasible

    active_thresh = 0.4 * rho_max
    recovered = set()
    for z in range(1, nz + 1):
        for c in range(nx):
            if grid[z][c] >= active_thresh:
                recovered.add((c, z))
    true_set = set((c, z_true) for c in range(c_lo, c_hi + 1))
    inter = len(recovered & true_set)
    union = len(recovered | true_set)
    iou = inter / union if union > 0 else 0.0

    stations = given + held
    errs = []
    for cs, r in stations:
        p = predict(grid, nx, nz, cs)
        if not math.isfinite(p):
            return None
        e = abs(p - r) / (abs(p) + abs(r) + 1e-9)
        errs.append(min(1.0, e))
    fitq = 1.0 - (sum(errs) / len(errs) if errs else 0.0)

    Q = W_IOU * iou + W_FIT * fitq
    return Q


def main():
    if len(sys.argv) < 3:
        out_ratio(0.0, "usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        test_id, nx, nz, rho_max, mass_max, given = read_instance(inf)
    except Exception:
        out_ratio(0.0, "bad instance")

    tb = true_body(test_id)
    if tb is None:
        out_ratio(0.0, "bad test id")
    tnx, tnz, z_true, c_lo, c_hi, rho_true, tmass_max = tb
    if tnx != nx or tnz != nz:
        out_ratio(0.0, "instance/testId mismatch")

    given_cols = [c for c in range(nx) if c % 2 == 0]
    held_cols = [c for c in range(nx) if c % 2 == 1]
    given_true = [(c, true_field(c, z_true, c_lo, c_hi, rho_true) * noise_factor(test_id, c))
                  for c in given_cols]
    held_true = [(c, true_field(c, z_true, c_lo, c_hi, rho_true) * noise_factor(test_id, c))
                 for c in held_cols]

    # -- parse submission --
    try:
        text = open(outf).read()
    except Exception:
        out_ratio(0.0, "cannot read output")
    toks = text.split()
    if len(toks) == 0:
        out_ratio(0.0, "empty output")
    if len(toks) > MAX_TOKENS:
        out_ratio(0.0, "too many tokens")
    if len(toks) != nx * nz:
        out_ratio(0.0, "expected %d tokens, got %d" % (nx * nz, len(toks)))
    vals = []
    for tok in toks:
        try:
            v = float(tok)
        except ValueError:
            out_ratio(0.0, "non-numeric token")
        if not math.isfinite(v):
            out_ratio(0.0, "non-finite value")
        if v < -1e-9 or v > rho_max + 1e-6:
            out_ratio(0.0, "value %.6g out of range [0,%.6g]" % (v, rho_max))
        vals.append(max(0.0, v))

    grid = [[0.0] * nx for _ in range(nz + 1)]
    p = 0
    for z in range(1, nz + 1):
        for c in range(nx):
            grid[z][c] = vals[p]; p += 1

    Q = score_grid(grid, nx, nz, rho_max, mass_max, z_true, c_lo, c_hi, rho_true,
                    given_true, held_true)
    if Q is None:
        out_ratio(0.0, "mass budget exceeded or non-finite prediction")

    # -- checker's internal baseline B (== solutions/trivial.py's construction) --
    tgrid = trivial_grid(nx, nz, rho_max, mass_max, given_true)
    B = score_grid(tgrid, nx, nz, rho_max, mass_max, z_true, c_lo, c_hi, rho_true,
                    given_true, held_true)
    B = max(B if B is not None else 0.0, 1e-6)

    ratio = min(CAP, 0.1 * Q / B)
    if ratio < 0.0:
        ratio = 0.0
    print("Q=%.4f B=%.4f" % (Q, B))
    print("Ratio: %.6f" % ratio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
