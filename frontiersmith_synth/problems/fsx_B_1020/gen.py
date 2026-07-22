#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of fsx_B_1020 to stdout.
Format:
  line 1: L T
  line 2: Jmax
  line 3: target_type   (0=engulfment/blob, 1=interleaved)
  line 4: n_0 n_1 n_2
  line 5: c_0 c_1 ... c_{L-1}   (initial ring arrangement, seeded shuffle)

All 10 cases are fully determined by testId (fixed table below, ascending
in L). target_type alternates 0,1,0,1,... so cases 2,4,6,8,10 (5 of 10,
well over the required >=3) are TRAP cases: the interleaved topology is
only reachable via FRUSTRATED (off-diagonal-favoring) adhesion; the
intuitive "like adheres to like" (diagonal-dominant) matrix drives the ring
toward full segregation instead -- the opposite of what these cases need,
so it lands far from what a frustration-aware strategy achieves.
"""
import sys, random

# (n0, n1, n2, Jmax, target_type)
TABLE = {
    1:  (3, 3, 3, 6, 0),
    2:  (3, 3, 3, 6, 1),
    3:  (4, 4, 4, 7, 0),
    4:  (4, 4, 4, 7, 1),
    5:  (5, 4, 3, 7, 0),
    6:  (5, 4, 3, 7, 1),
    7:  (6, 5, 4, 8, 0),
    8:  (6, 5, 4, 8, 1),
    9:  (8, 7, 6, 9, 0),
    10: (9, 7, 5, 9, 1),
}


def main():
    t = int(sys.argv[1])
    n0, n1, n2, Jmax, target_type = TABLE[t]
    counts = [n0, n1, n2]
    L = sum(counts)
    T = 3

    arr = []
    for i, c in enumerate(counts):
        arr += [i] * c
    rnd = random.Random(2000 + t)
    rnd.shuffle(arr)

    print(L, T)
    print(Jmax)
    print(target_type)
    print(n0, n1, n2)
    print(" ".join(str(x) for x in arr))


if __name__ == "__main__":
    main()
