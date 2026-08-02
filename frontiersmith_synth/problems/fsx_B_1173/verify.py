#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for the hydrophone TDOA localization task.

Feasibility: the participant's stdout must contain exactly K_test lines of "x y",
both finite floats, with |x|,|y| bounded generously (rejects nan/inf/garbage/huge).

Objective (maximize accuracy): the hidden ground-truth positions of the K_test
held-out emitters, and the per-receiver clock biases, are regenerated HERE by
literally duplicating gen.py's seeded construction (same test_id -> byte-identical
instance) -- they are never written to the .in file the solver reads.

Score per emitter: closeness(err) = D / (D + err), a bounded (0,1] function of the
Euclidean position error (err=0 -> 1; err>>D -> ~0). F = mean closeness over the
K_test emitters for the submission; B = the SAME formula evaluated at the checker's
own trivial baseline predictor (predict the receiver centroid X_ref for every held-
out emitter, ignoring all TDOA data). Ratio = min(1000, 100*F/max(1e-9,B))/1000.
A submission that reproduces the baseline predictor scores exactly 0.1.
"""
import sys
import math
import random

C_SPEED = 1481.0
AREA = 2000.0
BETA_MAX = 0.02
SIGMA_MEAS = 3.0e-2
K_CAL = 14
K_TEST = 10
JIT_LO, JIT_HI = 0.15, 0.5
SPREAD_ALONG = 1300.0
SPREAD_ACROSS = 40.0

TRAP_IDS = frozenset({2, 4, 6, 8, 10})
R_LADDER = [4, 4, 5, 5, 5, 6, 6, 6, 7, 7]

SEED_BASE = 1173000

D_SCALE = 100.0             # scoring closeness length-scale (meters)
POS_BOUND = 1.0e7           # sanity bound on submitted coordinates


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


# ---------- identical seeded construction to gen.py (hidden ground truth lives
# only here and in gen.py; never in the .in file) ----------
def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _gen_receivers(rng, R, trap):
    if trap:
        ang = rng.uniform(0.0, math.pi)
        ux, uy = math.cos(ang), math.sin(ang)
        vx, vy = -uy, ux
        cx, cy = AREA / 2.0, AREA / 2.0
        ts = sorted(rng.uniform(-AREA / 2.0, AREA / 2.0) for _ in range(R))
        recv = []
        for t in ts:
            jit = rng.uniform(JIT_LO, JIT_HI) * rng.choice((-1.0, 1.0))
            recv.append((cx + t * ux + jit * vx, cy + t * uy + jit * vy))
        return recv
    return [(rng.uniform(0.0, AREA), rng.uniform(0.0, AREA)) for _ in range(R)]


def _pca_axes(receivers):
    n = len(receivers)
    cx = sum(p[0] for p in receivers) / n
    cy = sum(p[1] for p in receivers) / n
    sxx = sum((p[0] - cx) ** 2 for p in receivers) / n
    syy = sum((p[1] - cy) ** 2 for p in receivers) / n
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in receivers) / n
    tr, diff = sxx + syy, sxx - syy
    disc = math.sqrt(max(0.0, diff * diff + 4 * sxy * sxy))
    lam1 = (tr + disc) / 2.0
    if abs(sxy) > 1e-12:
        e1 = (lam1 - syy, sxy)
        n1 = math.hypot(*e1)
        e1 = (e1[0] / n1, e1[1] / n1)
    else:
        e1 = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
    e2 = (-e1[1], e1[0])
    return (cx, cy), e1, e2


def _build_jacobian(receivers, x_ref):
    x0, y0 = receivers[0]
    d0 = _dist(x_ref, receivers[0])
    u0 = ((x_ref[0] - x0) / d0, (x_ref[1] - y0) / d0)
    J = []
    for (xr, yr) in receivers[1:]:
        dr = _dist(x_ref, (xr, yr))
        ur = ((x_ref[0] - xr) / dr, (x_ref[1] - yr) / dr)
        J.append(((ur[0] - u0[0]) / C_SPEED, (ur[1] - u0[1]) / C_SPEED))
    return J


def build(test_id):
    rng = random.Random(SEED_BASE + test_id * 7919)
    R = R_LADDER[(test_id - 1) % len(R_LADDER)]
    trap = test_id in TRAP_IDS
    receivers = _gen_receivers(rng, R, trap)
    beta = [rng.uniform(-BETA_MAX, BETA_MAX) for _ in range(R - 1)]
    x_ref, e_along, e_across = _pca_axes(receivers)
    J = _build_jacobian(receivers, x_ref)

    def make_offset():
        u = rng.uniform(-SPREAD_ALONG, SPREAD_ALONG)
        v = rng.uniform(-SPREAD_ACROSS, SPREAD_ACROSS)
        return (u * e_along[0] + v * e_across[0], u * e_along[1] + v * e_across[1])

    def tdoa(offset):
        m = R - 1
        base = [J[i][0] * offset[0] + J[i][1] * offset[1] for i in range(m)]
        return [base[i] + beta[i] + rng.gauss(0.0, SIGMA_MEAS) for i in range(m)]

    calib_off = [make_offset() for _ in range(K_CAL)]
    calib = [(off, tdoa(off)) for off in calib_off]
    test_off = [make_offset() for _ in range(K_TEST)]
    test_tau = [tdoa(off) for off in test_off]

    return dict(R=R, trap=trap, receivers=receivers, beta=beta, x_ref=x_ref, J=J,
                calib=calib, test_off=test_off, test_tau=test_tau)


def closeness(err, D=D_SCALE):
    return D / (D + err)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        with open(in_path) as fh:
            header = fh.readline().split()
        test_id = int(header[0])
    except Exception:
        fail("cannot parse test id from input header")
    if test_id < 1 or test_id > 100000:
        fail("bad test id")

    inst = build(test_id)
    x_ref = inst["x_ref"]
    true_pos = [(x_ref[0] + ox, x_ref[1] + oy) for (ox, oy) in inst["test_off"]]
    k_test = len(true_pos)

    try:
        with open(out_path, "rb") as fh:
            raw = fh.read(2_000_000)
    except Exception:
        fail("cannot read output")
    text = raw.decode("utf-8", "replace")
    toks = text.split()
    if len(toks) != 2 * k_test:
        fail("expected %d floats (%d lines of \"x y\"), got %d tokens" % (2 * k_test, k_test, len(toks)))

    submitted = []
    for i in range(k_test):
        try:
            xv = float(toks[2 * i])
            yv = float(toks[2 * i + 1])
        except ValueError:
            fail("non-numeric token at emitter %d" % i)
        if xv != xv or yv != yv or xv in (float("inf"), float("-inf")) or yv in (float("inf"), float("-inf")):
            fail("non-finite coordinate at emitter %d" % i)
        if abs(xv) > POS_BOUND or abs(yv) > POS_BOUND:
            fail("coordinate out of bounds at emitter %d" % i)
        submitted.append((xv, yv))

    f_vals = [closeness(_dist(submitted[i], true_pos[i])) for i in range(k_test)]
    b_vals = [closeness(_dist(x_ref, true_pos[i])) for i in range(k_test)]
    F = sum(f_vals) / k_test
    B = sum(b_vals) / k_test

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f test_id=%d" % (F, B, test_id))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
