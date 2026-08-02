# TIER: greedy
"""Textbook two-parameter linear regression: wait ~= (a + b*B^2) * L, fit by
ordinary least squares (normal equations) on features L and B^2*L. This is
the obvious "fit the visible trend" approach -- it captures the burstiness
interaction correctly, and on the training range (low-to-moderate load) it
looks like an excellent fit. But it is still LINEAR in L: it has no pole, so
it structurally cannot express the hyperbolic blow-up as load approaches
capacity, and it silently underestimates the held-out (higher-load) rows."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = []
    idx = 2
    for _ in range(n):
        L = float(data[idx]); B = float(data[idx + 1]); W = float(data[idx + 2])
        idx += 3
        rows.append((L, B, W))

    # features: f1 = L, f2 = B^2 * L ; solve normal equations for W ~= a*f1 + b*f2
    S11 = S12 = S22 = T1 = T2 = 0.0
    for L, B, W in rows:
        f1 = L
        f2 = (B * B) * L
        S11 += f1 * f1
        S12 += f1 * f2
        S22 += f2 * f2
        T1 += f1 * W
        T2 += f2 * W

    det = S11 * S22 - S12 * S12
    if abs(det) > 1e-9:
        a = (T1 * S22 - T2 * S12) / det
        b = (S11 * T2 - S12 * T1) / det
    else:
        a = T1 / S11 if S11 > 1e-9 else 0.0
        b = 0.0

    print("( %.6f + %.6f * B ** 2 ) * L" % (a, b))


if __name__ == "__main__":
    main()
