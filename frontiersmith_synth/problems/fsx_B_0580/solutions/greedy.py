# TIER: greedy
# Naive cheapest-to-convert knapsack: rank households by their own self-activation
# cost theta_i / r_i and pay each in full until the budget runs out. Ignores the
# network entirely -- treats every household's cost as independent.
import sys, math

def main():
    it = iter(sys.stdin.buffer.read().split())
    n = int(next(it)); m = int(next(it)); B = int(next(it)); W = int(next(it))
    theta = [0]*n; r = [0]*n
    for i in range(n):
        theta[i] = int(next(it)); r[i] = int(next(it))
    # (edges irrelevant to this heuristic)
    order = sorted(range(n), key=lambda i: (theta[i]/r[i], i))
    x = [0]*n; rem = B
    for i in order:
        c = -(-theta[i] // r[i])          # ceil(theta_i / r_i)
        if 0 < c <= rem:
            x[i] = c; rem -= c
    sys.stdout.write(" ".join(map(str, x)) + "\n")

main()
