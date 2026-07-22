#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of fsx_B_1028 to stdout.
Family: patrol-phase-cover (format C, minimize worst blind window).

Graph: a k-petal "flower" (windmill). Node 0 is the hub. Petal p is a simple
cycle of L_p nodes THROUGH the hub (L_p - 1 private nodes plus the shared
hub), so petals share only the hub. k in {2,3,4}; L_p >= 3.

Format:
  line 1: k P        (k = #petals, P = period cap for EACH guard's walk)
  line 2: L_1 ... L_k

All 10 cases are fully determined by testId (no external randomness).
Cases 2, 4, 7, 10 are TRAP cases: petal 1 and petal k have the SAME length
(k=3, palindromic) or all petals are equal -- the natural "send the guards
in opposite directions around the flower" fix leaves the MIDDLE petal's
relative timing untouched by the reversal (reversing a length-3 sequence
fixes its center element), so the two guards keep colliding there forever;
an approach that also searches per-guard periods/phases (not just route
direction) escapes this and does far better.
"""
import sys

# (k, [L_1..L_k], P)
# P is set to 2*sum(L)-2k+5: enough headroom above the checker's own
# "naive out-and-back per petal" baseline period (2*sum(L)-2k) for a
# solver to pad/rotate a full-loop tour (period sum(L)) without hitting cap.
TABLE = {
    1:  (2, [3, 5]),
    2:  (3, [3, 4, 3]),
    3:  (2, [7, 3]),
    4:  (3, [4, 4, 4]),
    5:  (2, [8, 4]),
    6:  (4, [4, 4, 3, 3]),
    7:  (3, [5, 4, 5]),
    8:  (2, [9, 5]),
    9:  (4, [5, 5, 3, 3]),
    10: (3, [6, 6, 6]),
}


def main():
    t = int(sys.argv[1])
    k, Ls = TABLE[t]
    total = sum(Ls)
    P = 2 * total - 2 * k + 5
    print(k, P)
    print(" ".join(str(x) for x in Ls))


if __name__ == "__main__":
    main()
