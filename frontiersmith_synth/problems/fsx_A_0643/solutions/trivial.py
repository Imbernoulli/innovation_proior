# TIER: trivial
"""
Trivial baseline: ignore Re entirely and predict the flat geometric mean of
the training Cd values everywhere. This reproduces the checker's own internal
baseline construction, so it should land at Ratio ~= 0.1.
"""
import sys, math


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    cds = []
    for _ in range(n):
        Re = float(data[idx]); idx += 1
        Cd = float(data[idx]); idx += 1
        cds.append(Cd)
    geo_mean = math.exp(sum(math.log(c) for c in cds) / len(cds))
    print("%.10g" % geo_mean)


if __name__ == "__main__":
    main()
