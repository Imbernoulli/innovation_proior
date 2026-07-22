# TIER: trivial
"""
Trivial baseline: ignore x entirely and predict the flat geometric mean of
the training READINGS (the raw, aliased frequency values) everywhere. This
reproduces the checker's own internal baseline construction exactly, so it
should land at Ratio ~= 0.1.
"""
import sys, math


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    fs = float(data[idx]); idx += 1
    fmax = float(data[idx]); idx += 1
    readings = []
    for _ in range(n):
        x = float(data[idx]); idx += 1
        r = float(data[idx]); idx += 1
        readings.append(max(1e-6, r))
    geo_mean = math.exp(sum(math.log(r) for r in readings) / len(readings))
    print("%.10g" % geo_mean)


if __name__ == "__main__":
    main()
