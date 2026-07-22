# TIER: greedy
# The obvious trap: treat it as an EXPECTED-VALUE knapsack.  Sort items by mean
# payoff per resource and fill both capacities.  Maximizes the average total but
# concentrates in high-mean HI items that all crash together at the low quantile.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); S = int(next(it)); W = int(next(it)); V = int(next(it)); k = int(next(it))
    w = [0] * N; v = [0] * N; mean = [0.0] * N
    for i in range(N):
        w[i] = int(next(it)); v[i] = int(next(it))
        s = 0
        for _ in range(S):
            s += int(next(it))
        mean[i] = s / S
    order = sorted(range(N), key=lambda i: -mean[i] / (w[i] + v[i]))
    sel = []
    cw = cv = 0
    for i in order:
        if cw + w[i] <= W and cv + v[i] <= V:
            sel.append(i); cw += w[i]; cv += v[i]
    sys.stdout.write(" ".join(str(i) for i in sel) + "\n")


main()
