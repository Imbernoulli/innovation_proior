#!/usr/bin/env python3
"""gen.py <testId>  ->  prints ONE training dataset for a geothermal-well scaling law.

The hidden ground-truth law lives here (and, independently, inside the checker
verify.py). STDOUT contains DATA ROWS ONLY plus a structural header
(`field_id n_train`) -- it never prints the law, its coefficients, or the noise
values. testId in 1..N selects the geothermal field (coefficients + noise seed).

Physical story: as the produced-fluid throughput scale x (dimensionless, ~ kg/s)
grows, the field's *specific thermal decline* y (deg C per MWh, a "loss"-like
quantity) falls off as a power law toward a nonzero irreducible floor E caused by
conductive recharge. Train rows sample the small-to-mid throughput regime; the
checker scores EXTRAPOLATION into the large-throughput regime (held out).
"""
import sys, math

N_TRAIN = 24
X_TRAIN_LO, X_TRAIN_HI = 10.0, 250.0
AMP = 0.14  # irreducible measurement scatter (multiplicative)


def _u01(a, b, c):
    """Deterministic pseudo-uniform in [0,1); no wall-time / RNG state."""
    x = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791)
    x &= 0x7FFFFFFF
    x = (x * 1103515245 + 12345) & 0x7FFFFFFF
    return x / 0x7FFFFFFF


def coeffs(t):
    """Ground-truth law parameters for field id t (kept in sync with verify.py)."""
    E = 2.0 + 0.3 * t     # irreducible floor
    A = 50.0 + 10.0 * t   # amplitude
    al = 0.5 + 0.03 * t   # decay exponent
    return E, A, al


def law(x, E, A, al):
    return E + A * x ** (-al)


def sample(t, x0, x1, nn, tag):
    E, A, al = coeffs(t)
    rows = []
    for i in range(nn):
        lx = math.log(x0) + (math.log(x1) - math.log(x0)) * i / (nn - 1)
        x = math.exp(lx)
        y = law(x, E, A, al) * (1.0 + AMP * (_u01(t, i, tag) - 0.5) * 2.0)
        rows.append((x, y))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = sample(t, X_TRAIN_LO, X_TRAIN_HI, N_TRAIN, 1)
    out = ["%d %d" % (t, len(rows))]
    for x, y in rows:
        out.append("%.10g %.10g" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
