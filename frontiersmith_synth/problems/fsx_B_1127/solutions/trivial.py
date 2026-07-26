# TIER: trivial
"""
Single fixed relaxation time (geometric mean of the observed lags).
This mirrors the checker's own internal baseline construction almost exactly
-- the simplest guess a novice would try: one exponential, no attempt to
discover any richer shape. Expected score ~0.1 by design.
"""
import sys
import math


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = []
    idx = 2
    for _ in range(n):
        g0 = float(data[idx]); tt = float(data[idx + 1]); sig = float(data[idx + 2])
        rows.append((g0, tt, sig))
        idx += 3

    ts = [r[1] for r in rows]
    tau = math.sqrt(min(ts) * max(ts))
    num = 0.0
    den = 0.0
    for g0, t, s in rows:
        x = math.exp(-t / tau)
        y = s / g0
        num += x * y
        den += x * x
    a = num / den if den > 1e-12 else 0.0

    print("%.8f * exp( -t / %.8f )" % (a, tau))


if __name__ == "__main__":
    main()
