# TIER: trivial
# Do-nothing baseline: pool ALL training readings into one global empirical
# variance about the known mean mu and predict that SAME constant everywhere,
# regardless of load x. This exactly reproduces the checker's own baseline
# construction -> Ratio ~= 0.1 by the scoring formula's design (matching the
# baseline gives 100*B/F = 100 -> ratio 0.1, not 1.0).
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 4:
        print("2.0")
        return
    n_x, R, mu = int(data[0]), int(data[1]), float(data[2])
    idx = 4
    ss = 0.0
    cnt = 0
    for _ in range(n_x):
        idx += 1  # skip x value
        for _ in range(R):
            y = float(data[idx]); idx += 1
            ss += (y - mu) ** 2
            cnt += 1
    var = ss / max(1, cnt)
    if var <= 0:
        var = 1e-3
    print("%.9f" % var)


if __name__ == "__main__":
    main()
