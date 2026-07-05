# TIER: trivial
# Predict a single constant: the mean clutch-rating of the training rows.
# This reproduces the checker's own baseline, so it scores ~0.1.
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.0")
        return
    n = int(data[0])
    # data layout: n, 4, then n*5 floats
    vals = data[2:]
    ys = []
    for r in range(n):
        base = r * 5
        if base + 4 < len(vals):
            ys.append(float(vals[base + 4]))
    m = sum(ys) / len(ys) if ys else 0.0
    print(repr(float(m)))


if __name__ == "__main__":
    main()
