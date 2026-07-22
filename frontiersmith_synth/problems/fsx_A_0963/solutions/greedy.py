# TIER: greedy
"""
The obvious "textbook" attempt: use the whole archive, not just half of it, and fit
one economy-wide flat marginal rate from total hours / total wages across ALL
training workers, then submit a single flat tax. This beats only skimming half the
scrolls, but it is still just a curve fit to the (wage,hours) data the solver was
given: the bunched workers are folded in as ordinary data points, not recognized as
a pile-up at a hidden threshold, so the fitted single rate is a same-shape-fits-all
compromise between the smooth region and the bunching-distorted region -- and it
carries no structural information about the never-observed upper bracket, only
whatever coincidental resemblance the average happens to have to it. It does not
reconstruct the notch, the threshold, or a second bracket at all.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    N = int(data[idx]); idx += 1
    W_LO = float(data[idx]); idx += 1
    W_HI = float(data[idx]); idx += 1
    dT = float(data[idx]); idx += 1
    sum_w = 0.0; sum_h = 0.0
    for _ in range(N):
        w = float(data[idx]); idx += 1
        h = float(data[idx]); idx += 1
        sum_w += w; sum_h += h
    tau_bar = 1.0 - sum_h / sum_w if sum_w > 0 else 0.2
    tau_bar = min(max(tau_bar, 0.0), 0.95)
    print("%.6f * z" % tau_bar)


if __name__ == "__main__":
    main()
