# TIER: greedy
# The obvious recipe: textbook root-sum-square (RSS) tolerance stacking.
# Assume every component's error is independent of every other's, so the
# total variance is just S2, and fit a single global scale correction c on
# D ~= c * sqrt(S2) by ordinary least squares. This matches the pilot-line
# logbook well -- batches there are tiny, so the correlated same-batch term
# is a small fraction of S2, comparable to sensor noise -- but the recipe
# never looks at the batch layout (m, B2) at all. On the held-out log,
# production has moved to a handful of LARGE batches and the correlated
# term overtakes S2; the RSS prediction falls far short of the truth.
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("1.0"); return
    rows = int(data[0])
    vals = data[2:]
    num = 0.0
    den = 0.0
    for i in range(rows):
        S2 = float(vals[5 * i + 2])
        D = float(vals[5 * i + 4])
        rss = math.sqrt(S2)
        num += D * rss
        den += rss * rss
    c = num / den if den > 1e-18 else 1.0
    print("%.10g * sqrt(S2)" % c)


if __name__ == "__main__":
    main()
