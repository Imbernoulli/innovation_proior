# TIER: greedy
"""The obvious "fix" once you notice that sharing the line evenly among every
pond starves everyone below their activation threshold: give the line to one
pond at a time. Split the horizon into P equal turns in the ponds' INPUT ORDER,
hand each pond the full cap C for its whole turn, and always use the entire
turn (no reason to stop early -- more steps can only add biomass). This is a
single competent pass: it solves the contention problem, but it never asks (a)
which pond should go FIRST -- a fast-decaying pond stuck in a late turn bleeds
value it never gets back, (b) whether two ponds could share a turn and shorten
the whole queue when their thresholds allow it, or (c) whether a pond's own
turn should be cut short before decay eats the marginal biomass it is still
adding."""
import sys
import math


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it))
    T = int(next(it))
    C = float(next(it))
    ponds = []
    for _ in range(P):
        a = float(next(it))
        b0 = float(next(it))
        e0 = float(next(it))
        decay = float(next(it))
        tau = float(next(it))
        ponds.append((a, b0, e0, decay, tau))

    W = T // P if P > 0 else 0
    out = [str(P)]
    for p in range(P):
        a, b0, e0, decay, tau = ponds[p]
        start = p * W
        end = start + W
        row = [f"{C:.6f}"] * W
        out.append(f"{start} {end}")
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
