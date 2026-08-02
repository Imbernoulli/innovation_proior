#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE hydrophone-array emitter-localization instance to stdout.

Story: a line of moored hydrophone buoys listens for a tagged transmitter's acoustic
pings. Each buoy has its own small, UNKNOWN clock offset (nuisance bias). The forward
model is the LINEARIZED hyperbolic time-difference-of-arrival (TDOA) relation around the
array centroid X_ref (arithmetic mean of the buoy positions):

    tau_r(X) = J_r . (X - X_ref) + beta_r + noise_r          (r = 1..R-1, receiver 0 is
                                                                the timing reference)

J_r (a 2-vector) is the EXACT gradient of the true hyperbolic TDOA equation
(|X-P_r|-|X-P_0|)/c evaluated at X_ref -- i.e. J_r = (u_r - u_0)/c where u_k is the unit
vector from X_ref toward receiver k. Everything an emitter can do to tau is captured by
this local-linear sensitivity; the array only ever hears targets within its operating
corridor (near the array), where this linearization is the forward model used throughout
(not merely an approximation used only when solving).

For a MINORITY of test ids the buoys are laid ALMOST exactly on one straight line (tiny
perpendicular jitter). Then J's 2x2 Gram matrix J^T J is severely ill-conditioned: one
combination of (x,y) offset from X_ref is data-starved (huge variance under measurement
noise) while the other is well pinned down. This is the "receiver-geometry-dilution"
trap: solving for BOTH coordinates from raw least squares amplifies noise catastrophically
along the starved axis.

Emitters (both calibration and held-out test emitters) are placed near X_ref along the
array's own principal axes: widely spread ALONG the array, narrowly spread ACROSS it --
so the calibration emitters' centroid is a genuinely informative prior for the
poorly-constrained axis, if a solver recognizes it needs one.

STDOUT (visible to the solver):
    R c
    R lines: "x y"                      (receiver/buoy positions, receiver 0 = reference)
    K_cal
    K_cal lines: "x y tau_1 ... tau_{R-1}"   (calibration emitter: KNOWN position + TDOA)
    K_test
    K_test lines: "tau_1 ... tau_{R-1}"      (held-out emitter: TDOA only -- position unknown)

The true held-out positions and the per-receiver clock biases are NEVER printed; they are
regenerated inside verify.py from the identical seeded construction below.
"""
import sys
import math
import random

C_SPEED = 1481.0          # m/s, fixed speed of sound in the operating medium
AREA = 2000.0              # receiver deployment footprint (meters)
BETA_MAX = 0.02            # s, per-receiver relative clock bias range (+-20 ms)
SIGMA_MEAS = 3.0e-2        # s, per-measurement TDOA noise std
K_CAL = 14                 # calibration emitters (known position)
K_TEST = 10                # held-out test emitters (position withheld, graded)
JIT_LO, JIT_HI = 0.15, 0.5   # perpendicular jitter (meters) for near-collinear trap arrays
SPREAD_ALONG = 1300.0       # emitter offset spread along the array's long axis (meters)
SPREAD_ACROSS = 40.0        # emitter offset spread across the array's short axis (meters)

TRAP_IDS = frozenset({2, 4, 6, 8, 10})
R_LADDER = [4, 4, 5, 5, 5, 6, 6, 6, 7, 7]

SEED_BASE = 1173000


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
    """Principal axes of the receiver point cloud (for shaping the emitter corridor
    only -- the solver never needs to reproduce this)."""
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
    """J_r = gradient of the true hyperbolic TDOA (|X-P_r|-|X-P_0|)/c at x_ref."""
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
    """Deterministic full instance construction (duplicated verbatim in verify.py so the
    checker can regenerate the hidden held-out truth and clock biases from test_id alone)."""
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


def render(test_id):
    inst = build(test_id)
    R = inst["R"]
    x_ref = inst["x_ref"]
    lines = ["%d %d %.3f" % (test_id, R, C_SPEED)]
    for (x, y) in inst["receivers"]:
        lines.append("%.6f %.6f" % (x, y))
    lines.append(str(K_CAL))
    for (off, taus) in inst["calib"]:
        ax, ay = x_ref[0] + off[0], x_ref[1] + off[1]
        lines.append("%.6f %.6f " % (ax, ay) + " ".join("%.9f" % t for t in taus))
    lines.append(str(K_TEST))
    for taus in inst["test_tau"]:
        lines.append(" ".join("%.9f" % t for t in taus))
    return "\n".join(lines) + "\n"


def main():
    test_id = int(sys.argv[1])
    sys.stdout.write(render(test_id))


if __name__ == "__main__":
    main()
