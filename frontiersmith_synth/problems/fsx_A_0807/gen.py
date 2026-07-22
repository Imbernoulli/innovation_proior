#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE calibration notebook to stdout.

A bench measures a response y that depends on four positive knobs
x1..x4. Each knob carries a fixed integer "grading" across 3 abstract
calibration axes (a 3x4 integer matrix U, given in the input) and the
response y is REQUIRED to be grading-neutral (axis-invariant). That
constraint forces the null space of U (as a linear map R^4 -> R^3) to
be exactly 1-dimensional, spanned by a unique (up to scale) integer
vector b: any grading-neutral formula for y must, in log space, have
its x_i-exponents proportional to b. So the hidden bench law is

    y = C * Pi ** p,     Pi = prod_i x_i ** b_i     (the ONE dimensionless group)

with unknown constants C (amplitude) and p (exponent). The TRAIN
notebook sweeps each x_i independently over a NARROW multiplicative
band around its own center -- the campaign never explores a wide
dynamic range in any single raw knob. The HELD-OUT grading grid
(regenerated ONLY inside the grader) pushes the knobs far outside that
band, mostly along directions that are a mix of the true Pi-direction
AND large "wasted" motion orthogonal to it (which raw-per-knob fits
cannot tell apart from real signal, but has zero effect on the true y).

STDOUT prints ONLY: header "<n> <test_id>", then 3 rows of the grading
matrix U (4 ints each), then n rows "x1 x2 x3 x4 y". The hidden b, p,
C, and seeds are NEVER printed -- only the grading matrix (needed by
the solver to run its own dimensional analysis) and the noisy data.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
BASE_SEED_PARAMS = 730100
MULT_PARAMS = 91121
BASE_SEED_TRAIN = 550301
MULT_TRAIN = 4021

W_TRAIN = math.log(1.30)     # +/-30% multiplicative training window per knob
NOISE_TRAIN = 0.04
N_TRAIN = 20

M_LO, M_HI = 2.0, 9.0
P_ABS_LO, P_ABS_HI = 0.5, 2.0
C_LO, C_HI = 0.8, 3.5


def _det3(R):
    return (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
            - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
            + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))


def params(t):
    """Hidden grading matrix + scaling law for this test id (identical in
    gen.py and verify.py)."""
    rng = random.Random(BASE_SEED_PARAMS + t * MULT_PARAMS)

    while True:
        b = [rng.randint(-2, 2) for _ in range(4)]
        if all(v != 0 for v in b):
            break
    b4 = b[3]

    while True:
        R = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(3)]
        if _det3(R) != 0:
            break

    U = []
    for j in range(3):
        r = R[j]
        S = r[0] * b[0] + r[1] * b[1] + r[2] * b[2]
        U.append([b4 * r[0], b4 * r[1], b4 * r[2], -S])

    m = [rng.uniform(M_LO, M_HI) for _ in range(4)]

    p_mag = rng.uniform(P_ABS_LO, P_ABS_HI)
    p_mag = round(p_mag * 4) / 4.0
    if p_mag < 0.25:
        p_mag = 0.25
    p = p_mag if rng.random() < 0.5 else -p_mag

    C = rng.uniform(C_LO, C_HI)
    return U, b, m, p, C


def pi_group(x, b):
    val = 1.0
    for xi, bi in zip(x, b):
        val *= xi ** bi
    return val


def true_y(x, b, p, C):
    return C * (pi_group(x, b) ** p)


def gen_train(t):
    U, b, m, p, C = params(t)
    rng = random.Random(BASE_SEED_TRAIN + t * MULT_TRAIN)
    rows = []
    for _ in range(N_TRAIN):
        x = [m[i] * math.exp(rng.uniform(-W_TRAIN, W_TRAIN)) for i in range(4)]
        y = true_y(x, b, p, C) * math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((x, y))
    return rows, U


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows, U = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for row in U:
        out.append(" ".join(str(v) for v in row))
    for x, y in rows:
        out.append("%.8g %.8g %.8g %.8g %.8g" % (x[0], x[1], x[2], x[3], y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
