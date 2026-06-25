#!/usr/bin/env python3
# Independent brute force for the "Cheapest pace relay" problem.
#
# n stones in a line (indices 0..n-1). From stone i you may hop:
#   +1 : to stone i+1, with effort e1[i], distance g1[i]   (defined for i=0..n-2)
#   +2 : to stone i+2, with effort e2[i], distance g2[i]   (defined for i=0..n-3)
# A plan is any sequence of hops starting at stone 0 and ending exactly at stone n-1.
# Its pace = (sum of efforts of used hops) / (sum of distances of used hops).
# Output the MINIMUM possible pace over all plans, as a reduced fraction "P Q" (Q>0).
#
# Brute force: enumerate EVERY hop-path 0 -> n-1 by recursion (+1 / +2), compute exact
# pace with Fraction, keep the minimum.
import sys
from fractions import Fraction
from math import gcd

def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    e1 = [0]*(n-1); g1 = [0]*(n-1)
    for i in range(n-1):
        e1[i] = int(data[idx]); idx += 1
        g1[i] = int(data[idx]); idx += 1
    e2 = [0]*max(0, n-2); g2 = [0]*max(0, n-2)
    for i in range(n-2):
        e2[i] = int(data[idx]); idx += 1
        g2[i] = int(data[idx]); idx += 1

    best = [None]  # (E, D) with minimal E/D

    def consider(E, D):
        if best[0] is None:
            best[0] = (E, D)
        else:
            bE, bD = best[0]
            # E/D < bE/bD  <=>  E*bD < bE*D  (all denominators positive)
            if E * bD < bE * D:
                best[0] = (E, D)

    # iterative DFS over positions to avoid recursion-limit issues
    stack = [(0, 0, 0)]  # (position, accumulated E, accumulated D)
    while stack:
        pos, E, D = stack.pop()
        if pos == n-1:
            consider(E, D)
            continue
        if pos == n:
            continue
        # +1 hop
        if pos+1 <= n-1:
            stack.append((pos+1, E + e1[pos], D + g1[pos]))
        # +2 hop
        if pos+2 <= n-1:
            stack.append((pos+2, E + e2[pos], D + g2[pos]))

    E, D = best[0]
    g = gcd(E, D)
    print(E // g, D // g)

if __name__ == "__main__":
    main()
