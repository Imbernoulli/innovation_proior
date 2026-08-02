# TIER: trivial
# Do-nothing baseline: an ordinary least-squares STRAIGHT LINE y = a + b*x fit
# to the training rows, ignoring the dopant descriptors entirely.  This is
# exactly the checker's own internal baseline construction, so it reproduces
# Ratio ~= 0.1 by design.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[1])
    xs = []
    ys = []
    for i in range(n):
        base = 2 + 5 * i
        xs.append(float(data[base + 1]))
        ys.append(float(data[base + 4]))

    Sx = sum(xs)
    Sy = sum(ys)
    Sxx = sum(v * v for v in xs)
    Sxy = sum(u * v for u, v in zip(xs, ys))
    denom = n * Sxx - Sx * Sx
    b = (n * Sxy - Sx * Sy) / denom if abs(denom) > 1e-12 else 0.0
    a = (Sy - b * Sx) / n

    print("(%.6f) + (%.6f) * x" % (a, b))


if __name__ == "__main__":
    main()
