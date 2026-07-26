# TIER: greedy
# The obvious recipe: fit a single straight line y = a + b*t to the raw
# training days by ordinary least squares, ignoring the seasonal eddy and
# the rip-current growth term entirely. This tracks the AVERAGE drift over
# the training window reasonably, but a fixed slope cannot bend to match the
# accelerating cubic term -- it increasingly under/over-shoots as the graded
# window runs on for twice the training horizon.
import sys


def main():
    data = sys.stdin.read().split()
    T_train = int(data[0])
    ts, ys = [], []
    idx = 5
    for _ in range(T_train):
        i = int(data[idx]); obs = int(data[idx + 1]); idx += 2
        ts.append(float(i)); ys.append(float(obs))

    n = len(ts)
    mean_t = sum(ts) / n
    mean_y = sum(ys) / n
    sxy = sum((t - mean_t) * (y - mean_y) for t, y in zip(ts, ys))
    sxx = sum((t - mean_t) ** 2 for t in ts)
    b = sxy / sxx if sxx > 1e-9 else 0.0
    a = mean_y - b * mean_t

    print("EXPR %.10f + %.10f * t" % (a, b))


if __name__ == "__main__":
    main()
