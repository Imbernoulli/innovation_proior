# TIER: greedy
# The "obvious" first idea: a bank's own solvency gap (what it owes minus its
# own cash, ignoring anything the network might pay it) measures how badly it
# needs help. Rescue the worst-off banks first, fully funding each one before
# moving to the next, until the budget runs out. This is a natural, honest
# recipe -- but it never asks whether saving a bank helps anyone ELSE.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); c = int(next(it))
    e = [float(next(it)) for _ in range(n)]
    m = int(next(it))
    pbar = [0.0] * n
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = float(next(it))
        pbar[u] += w

    shortfall = [max(0.0, pbar[i] - e[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -shortfall[i])

    delta = [0.0] * n
    remaining = float(c)
    for i in order:
        if remaining <= 1e-9:
            break
        need = shortfall[i]
        if need <= 1e-9:
            continue
        give = min(remaining, need)
        delta[i] = give
        remaining -= give

    print(" ".join("%.6f" % x for x in delta))

main()
