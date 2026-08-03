#!/usr/bin/env python3
"""Generator for fsx_B_0172 -- harbor container-port turnaround scaling law.

`python3 gen.py <testId>` prints ONE train sample to stdout.

Only DATA ROWS are printed. The hidden ground-truth law, its coefficients and the
noise seed are NEVER printed -- the solver must discover the functional form from
the (n, c, T) rows alone. The held-out EXTRAPOLATION split (much larger vessel
workloads / crane counts) lives only inside the checker.
"""
import sys, math, random

# ---- hidden ground-truth scaling law (server-side; never emitted to stdout) ----
T0, A, P, Q = 1.8, 0.045, 1.15, 0.85
def true_T(n, c):
    return T0 + A * (n ** P) / (c ** Q)


def main():
    tid = int(sys.argv[1])
    if tid < 1:
        tid = 1
    # difficulty ladder: higher testId -> more rows BUT more observation noise.
    M = 60 + 14 * tid
    s = 0.08 + 0.005 * tid          # multiplicative noise half-width on TRAIN
    rng = random.Random(1000 + tid)

    rows = []
    for _ in range(M):
        # train region: SMALL vessels, FEW cranes (log-uniform workload)
        n = math.exp(rng.uniform(math.log(200.0), math.log(2000.0)))
        c = rng.choice([2, 3, 4, 5])
        t = true_T(n, c) * (1.0 + rng.uniform(-s, s))
        rows.append((n, c, t))

    out = [str(M)]
    for n, c, t in rows:
        out.append("%.4f %d %.5f" % (n, c, t))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
