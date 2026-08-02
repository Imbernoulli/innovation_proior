#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Family: band-gap-extrapolate.  A hidden host semiconductor of band gap E0 is
doped by a small VISIBLE family of trace elements; each element is described
by its electronegativity mismatch to the host (dEN) and its covalent-radius
mismatch (dR), both FIXED per element.  For each element the logger swept the
doping fraction x across a narrow visible composition window and recorded the
resulting band gap y, with sensor noise.

The true law used to build y (E0, k1, k0, k4, k3 below) is NEVER printed --
only the (x, dEN, dR, y) rows are.  It also drives the checker's held-out
extrapolation split (composition range pushed further AND brand-new dopant
elements whose dEN/dR fall well outside anything seen here), which is
regenerated independently inside verify.py using the same testId.

STDOUT format:
    <testId> <n_rows>
    <dopant_idx> <x> <dEN> <dR> <y>      (n_rows lines)
"""
import sys
import random

X_MAX_VISIBLE = 0.15
SIGMA_FRAC = 0.018

# (n_dopants, points_per_dopant, dEN_half_range, dR_half_range) -- small -> large/adversarial
TRAIN_LADDER = {
    1: (6, 5, 0.35, 0.10),
    2: (6, 6, 0.35, 0.10),
    3: (7, 6, 0.32, 0.09),
    4: (7, 6, 0.32, 0.09),
    5: (8, 6, 0.30, 0.08),
    6: (8, 7, 0.30, 0.08),
    7: (9, 7, 0.28, 0.075),
    8: (9, 7, 0.26, 0.07),
    9: (10, 7, 0.24, 0.065),
    10: (10, 8, 0.20, 0.06),
}


def hidden_params(test_id):
    """Hidden materials constants for this test id (lives in gen AND checker,
    never printed).  E0=intrinsic host gap, k1/k0=composition (Vegard+bowing)
    terms, k4=electronegativity-NONLINEARITY term, k3=radius-mismatch term."""
    rng = random.Random(500000 + 97 * test_id)
    E0 = round(rng.uniform(1.6, 3.0), 4)
    k1 = round(rng.uniform(2.0, 5.0), 4)
    k0 = round(rng.uniform(16.0, 30.0), 4)
    k4 = round(rng.uniform(3.0, 8.0), 4)
    k3 = round(rng.uniform(3.0, 7.0), 4)
    return E0, k1, k0, k4, k3


def visible_dopants(test_id):
    """The VISIBLE chemistry family: n_dop elements with fixed (dEN, dR)."""
    n_dop, pts, den_half, dr_half = TRAIN_LADDER[test_id]
    rng = random.Random(707000 + 131 * test_id)
    dopants = []
    for _ in range(n_dop):
        dEN = round(rng.uniform(-den_half, den_half), 5)
        dR = round(rng.uniform(-dr_half, dr_half), 5)
        dopants.append((dEN, dR))
    return dopants


def true_y(params, x, dEN, dR):
    E0, k1, k0, k4, k3 = params
    return E0 - k1 * x - k0 * x * x - k4 * x * (dEN ** 2) - k3 * x * dR


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    params = hidden_params(test_id)
    E0 = params[0]
    dopants = visible_dopants(test_id)
    n_dop, pts, den_half, dr_half = TRAIN_LADDER[test_id]
    sigma = SIGMA_FRAC * E0
    noise_rng = random.Random(919000 + 17 * test_id)

    rows = []
    for idx, (dEN, dR) in enumerate(dopants):
        for j in range(pts):
            x = X_MAX_VISIBLE * j / (pts - 1) if pts > 1 else X_MAX_VISIBLE
            x = round(x, 6)
            y = true_y(params, x, dEN, dR) + noise_rng.gauss(0.0, sigma)
            rows.append((idx, x, dEN, dR, round(y, 6)))

    out = ["%d %d" % (test_id, len(rows))]
    for idx, x, dEN, dR, y in rows:
        out.append("%d %.6f %.6f %.6f %.6f" % (idx, x, dEN, dR, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
