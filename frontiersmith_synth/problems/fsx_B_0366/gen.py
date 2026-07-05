#!/usr/bin/env python3
"""Generator for fsx_B_0366 -- lunar-habitat life-support power scaling law.

`python3 gen.py <testId>` prints ONE train sample to stdout.

Only DATA ROWS are printed. The hidden ground-truth law, its coefficients and the
noise seed are NEVER printed -- the solver must discover the functional form from
the (k, V, P) rows alone. The held-out EXTRAPOLATION split (much larger crews /
habitat volumes) lives only inside the checker.
"""
import sys, math, random

# ---- hidden ground-truth scaling law (server-side; never emitted to stdout) ----
#   P = P0 + A * k^ALPHA * V^BETA
# superlinear in crew (metabolic + CO2 scrubbing), sublinear in volume (thermal/leak),
# plus a fixed baseline systems overhead P0.
P0, A, ALPHA, BETA = 0.5, 0.15, 1.2, 0.6
def true_P(k, V):
    return P0 + A * (k ** ALPHA) * (V ** BETA)


def main():
    tid = int(sys.argv[1])
    if tid < 1:
        tid = 1
    # difficulty ladder: higher testId -> more rows BUT more telemetry noise.
    M = 70 + 14 * tid
    s = 0.08 + 0.005 * tid          # multiplicative noise half-width on TRAIN
    rng = random.Random(2600 + tid)

    rows = []
    for _ in range(M):
        # train region: SMALL habitats -- few crew, modest volume (log-uniform V)
        k = rng.randint(2, 12)
        V = math.exp(rng.uniform(math.log(30.0), math.log(300.0)))
        p = true_P(k, V) * (1.0 + rng.uniform(-s, s))
        rows.append((k, V, p))

    out = [str(M)]
    for k, V, p in rows:
        out.append("%d %.4f %.5f" % (k, V, p))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
