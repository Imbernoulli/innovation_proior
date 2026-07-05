#!/usr/bin/env python3
"""
gen.py <testId>  -- prints the TRAIN sample for one instance of the
grid-efficiency scaling-law extrapolation problem.

Each line is a data row:  "<x> <eta>"
where x is the grid interconnection scale (number of coupled substations)
and eta is the measured aggregate delivery efficiency at that scale.

The hidden ground-truth law, its coefficients, the random seed and the
held-out (large-scale) region are NOT emitted here -- data rows only.
They live exclusively inside the checker (verify.py).
"""
import sys
import math


# ---- deterministic PRNG (LCG); identical logic must live in verify.py ----
def _rng(seed):
    state = [(seed * 2654435761 + 12345) & 0x7FFFFFFF]

    def nxt():
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        return state[0] / 0x7FFFFFFF

    return nxt


def derive_params(test_id):
    """Ground-truth coefficients for  eta(x) = c - a * x**(-b)  (saturating)."""
    r = _rng(1000 + test_id)
    c = 0.90 + 0.08 * r()   # asymptotic ceiling efficiency in [0.90, 0.98]
    a = 0.30 + 0.40 * r()   # size of the small-scale penalty in [0.30, 0.70]
    b = 0.35 + 0.30 * r()   # scaling exponent in [0.35, 0.65]
    return c, a, b


def noise_rel(test_id):
    # difficulty ladder: later tests carry more irreducible measurement noise
    return 0.012 + 0.004 * (test_id - 1)


def train_x(n=24, lo=8.0, hi=250.0):
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


def make_train(test_id):
    c, a, b = derive_params(test_id)
    xs = train_x()
    nr = noise_rel(test_id)
    rn = _rng(5000 + test_id)   # train-noise stream
    rows = []
    for x in xs:
        clean = c - a * x ** (-b)
        y = clean * (1.0 + nr * (2.0 * rn() - 1.0))
        rows.append((x, y))
    return rows


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    rows = make_train(test_id)
    out = []
    for x, y in rows:
        out.append("%.10g %.10g" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
