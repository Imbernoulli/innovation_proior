# TIER: trivial
"""
Trivial baseline: ignore T entirely and predict the flat geometric mean of
the training rise-rate values everywhere. This reproduces the checker's own
internal baseline construction, so it should land at Ratio ~= 0.1.
"""
import sys, math


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    rs = []
    for _ in range(n):
        T = float(data[idx]); idx += 1
        r = float(data[idx]); idx += 1
        rs.append(r)
    geo_mean = math.exp(sum(math.log(r) for r in rs) / len(rs))
    print("%.10g" % geo_mean)


if __name__ == "__main__":
    main()
