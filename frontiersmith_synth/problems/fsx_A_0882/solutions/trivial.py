# TIER: trivial
"""
Trivial baseline: ignore x entirely and predict the flat geometric mean of
the training y values everywhere. This reproduces the checker's own internal
baseline construction, so it should land at Ratio ~= 0.1.
"""
import sys, math


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    ys = []
    for _ in range(n):
        x = float(data[idx]); idx += 1
        y = float(data[idx]); idx += 1
        ys.append(y)
    geo_mean = math.exp(sum(math.log(y) for y in ys) / len(ys))
    print("%.10g" % geo_mean)


if __name__ == "__main__":
    main()
