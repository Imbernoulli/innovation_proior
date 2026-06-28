#!/usr/bin/env python3
"""
Independent brute-force oracle for q1(N) (Renyi-Ulam, one lie).

Method: the textbook EXHAUSTIVE game-tree recursion, with no closed form at all.
A state is (b0, b1): b0 candidates with 0 lies charged, b1 with 1 lie charged.
canwin(b0, b1, q) = "questioner can guarantee identification within q questions".

  - if b0+b1 <= 1: already identified -> True.
  - if q == 0: True iff <=1 candidate remains.
  - otherwise: the questioner picks how many of each level go into the "yes" set,
    (y0 in 0..b0, y1 in 0..b1). The adversary then answers yes or no, whichever
    is worse for us:
        YES  -> "no"-set candidates each gain a lie:
                   (b0 - y0) move from level 0 to level 1;
                   (b1 - y1) would reach level 2 -> eliminated.
                 new state (y0, y1 + (b0 - y0))
        NO   -> "yes"-set candidates each gain a lie:
                 new state (b0 - y0, (b1 - y1) + y0)
    The question works iff BOTH children are winnable in q-1. The questioner wins
    if SOME (y0,y1) makes that true.

q1(N) = least q with canwin(N, 0, q). This tries every split exhaustively, so it
is obviously correct; it is exponential and used only on small N.
"""
import sys
from functools import lru_cache

sys.setrecursionlimit(10_000_000)

@lru_cache(maxsize=None)
def canwin(b0, b1, q):
    if b0 + b1 <= 1:
        return True
    if q == 0:
        return b0 + b1 <= 1
    for y0 in range(b0 + 1):
        for y1 in range(b1 + 1):
            # YES branch: no-set gains a lie
            yb0, yb1 = y0, y1 + (b0 - y0)
            # NO branch: yes-set gains a lie
            nb0, nb1 = b0 - y0, (b1 - y1) + y0
            if canwin(yb0, yb1, q - 1) and canwin(nb0, nb1, q - 1):
                return True
    return False

def q1(N):
    if N <= 1:
        return 0
    q = 1
    while not canwin(N, 0, q):
        q += 1
    return q

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    T = int(data[idx]); idx += 1
    out = []
    for _ in range(T):
        N = int(data[idx]); idx += 1
        out.append(str(q1(N)))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

if __name__ == "__main__":
    main()
