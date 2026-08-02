#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE systolic-array dataflow-mapping instance to stdout.

Format:
  P Q L RELOAD SWITCH
  M_1 K_1 N_1
  ...
  M_L K_L N_L

testId 1..10 is a hand-authored difficulty/trap ladder:
  1-2: near-uniform layer shapes (a single fixed dataflow is close to optimal --
       sanity check that the objective doesn't reward switching for its own sake).
  3-10: mixed-shape sequences -- one shape class ("M-heavy": M >> K,N; "K-heavy";
       "N-heavy") dominates total volume, with several small layers drawn from the
       OTHER two shape classes. A fixed dataflow tuned for the dominant shape (or a
       single generic default) wastes the array on the minority-shape layers; the
       per-layer choice does not.
"""
import sys


def Mh(big, s1, s2=None):
    return (big, s1, s2 if s2 is not None else s1)


def Kh(big, s1, s2=None):
    return (s1, big, s2 if s2 is not None else s1)


def Nh(big, s1, s2=None):
    return (s1, s2 if s2 is not None else s1, big)


CASES = {
    1: dict(P=8, Q=8, RELOAD=4, SWITCH=6,
            layers=[(18, 18, 18), (22, 22, 22), (16, 16, 16)]),
    2: dict(P=6, Q=10, RELOAD=3, SWITCH=8,
            layers=[(24, 24, 24), (24, 24, 24), (24, 24, 24), (30, 20, 26)]),
    3: dict(P=6, Q=12, RELOAD=2, SWITCH=20,
            layers=[Kh(250, 5)] + [Mh(70, 6)] * 2 + [Nh(70, 6)] * 2),
    4: dict(P=8, Q=12, RELOAD=4, SWITCH=2,
            layers=[Kh(300, 5)] + [Mh(50, 6)] * 2 + [Nh(50, 6)] * 2),
    5: dict(P=12, Q=8, RELOAD=6, SWITCH=2,
            layers=[Nh(250, 4)] + [Mh(40, 8)] * 3 + [Kh(40, 8)] * 3),
    6: dict(P=6, Q=16, RELOAD=4, SWITCH=20,
            layers=[Mh(100, 3)] + [Kh(70, 6)] * 4 + [Nh(70, 6)] * 4),
    7: dict(P=6, Q=16, RELOAD=4, SWITCH=8,
            layers=[Kh(300, 3)] + [Mh(40, 4)] * 2 + [Nh(40, 4)] * 2),
    8: dict(P=16, Q=6, RELOAD=6, SWITCH=15,
            layers=[Nh(300, 4)] + [Mh(50, 5)] * 3 + [Kh(50, 5)] * 3),
    9: dict(P=8, Q=10, RELOAD=5, SWITCH=10,
            layers=[Mh(100, 4)] + [Kh(90, 6)] * 5 + [Nh(90, 6)] * 4),
    10: dict(P=10, Q=12, RELOAD=5, SWITCH=10,
             layers=[Mh(100, 4)] + [Kh(90, 5)] * 3 + [Nh(90, 5)] * 3),
}


def main():
    tid = int(sys.argv[1])
    tid = ((tid - 1) % 10) + 1
    c = CASES[tid]
    out = [f"{c['P']} {c['Q']} {len(c['layers'])} {c['RELOAD']} {c['SWITCH']}"]
    for (m, k, n) in c['layers']:
        out.append(f"{m} {k} {n}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
