# TIER: trivial
"""Naive proportional queueing model: wait ~= k*L, ignoring burstiness
entirely, fit through the origin via ordinary least squares on the training
rows. This is exactly the checker's own internal baseline construction, so
it reproduces Ratio ~= 0.10."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    # data[1] is the test id, then n triples (L, B, W)
    rows = []
    idx = 2
    for _ in range(n):
        L = float(data[idx]); B = float(data[idx + 1]); W = float(data[idx + 2])
        idx += 3
        rows.append((L, B, W))

    num = sum(L * W for L, B, W in rows)
    den = sum(L * L for L, B, W in rows)
    k = num / den if den > 1e-12 else 0.0

    print("%.6f * L" % k)


if __name__ == "__main__":
    main()
