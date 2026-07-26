# TIER: trivial
"""
Trivial baseline: ignore d and T entirely and predict the flat geometric
mean of the training breakdown-voltage values everywhere. This reproduces
the checker's own internal baseline construction, so it should land at
Ratio ~= 0.1.
"""
import sys
import math


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    vs = []
    for _ in range(n):
        d = float(data[idx]); idx += 1
        T = float(data[idx]); idx += 1
        v = float(data[idx]); idx += 1
        vs.append(v)
    geo_mean = math.exp(sum(math.log(v) for v in vs) / len(vs))
    print("%.10g" % geo_mean)


if __name__ == "__main__":
    main()
