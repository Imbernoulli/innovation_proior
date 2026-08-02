# TIER: greedy
# The obvious recipe: average the visible pre-treatment measurements (a
# sensible noise-reduced estimate of the current resistant frequency) and
# extrapolate that value FLAT into the future. This is the natural first
# instinct -- "no trend visible in the training window, so predict no
# change" -- and it is an honest, well-noise-reduced read of the DATA. It is
# blind to the mechanism: it never asks why the frequency is where it is, so
# it cannot see that the dosing plan (given in the input) is about to
# overturn the fitness-cost/selection balance and drive a sweep.
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    _t = int(data[idx]); idx += 1
    _mu, _tau, _alpha, _D, _T0, _T1 = (float(x) for x in data[idx:idx + 6]); idx += 6
    n_train = int(data[idx]); idx += 1
    obs = []
    for _ in range(n_train):
        _tt = float(data[idx]); idx += 1
        yy = float(data[idx]); idx += 1
        obs.append(yy)
    p_mean = sum(obs) / len(obs) if obs else 0.5
    print("%.6f" % p_mean)


if __name__ == "__main__":
    main()
