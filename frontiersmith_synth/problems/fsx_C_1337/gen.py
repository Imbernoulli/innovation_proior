#!/usr/bin/env python3
"""gen.py <testId> -- print ONE heat-exchanger-network instance to stdout.

Instances are a FIXED function of testId (1..10, difficulty ladder small->large),
so they are bit-for-bit reproducible without needing a PRNG. Each instance plants
a "pinch trap": a globally-hottest hot stream and a globally-coldest cold stream
that look maximally attractive to match directly (huge local temperature driving
force -> tiny required area) but whose capacities are needed elsewhere by streams
that have very few other feasible partners. Bigger testIds compose 2-3 such traps
plus flexible filler streams -> more streams, richer combinatorics.

Format (stdin, consumed by the participant's program):
  line 1: NH NC
  line 2: DTMIN CH CC A
  next NH lines: THs THt CPh      (hot stream: supply, target, heat-capacity flow; THs>THt)
  next NC lines: TCs TCt CPc      (cold stream: supply, target, heat-capacity flow; TCt>TCs)
"""
import sys


def instance(tid):
    dtmin, cH, cC, a = 10.0, 8.0, 6.0, 20.0

    # ---- reusable stream-pair templates (see notes above each) ----
    # trap group T1: H1 is the hottest hot stream, C1 the coldest cold stream.
    # H1-C1 is gate-feasible with a huge driving force (tempting), but C1 is the
    # ONLY cold stream H2 can reach, and C2 is the ONLY cold stream H1 can fully
    # discharge into without spare utility. Greedy (hottest-hot x coldest-cold)
    # steals C1 for H1 and strands H2 and the tail of C2.
    T1H = [(200.0, 140.0, 1.0), (120.0, 60.0, 1.0)]
    T1C = [(30.0, 90.0, 1.0), (110.0, 180.0, 1.0)]
    # trap group T2: same shape, independent numbers (different temperatures/CP
    # so it is not a re-skin of T1 within one instance).
    T2H = [(250.0, 210.0, 1.0), (100.0, 70.0, 1.0)]
    T2C = [(20.0, 70.0, 1.0), (230.0, 245.0, 1.0)]
    # flexible filler: an ordinary, unambiguous pair (always worth matching).
    FILLH = [(180.0, 140.0, 1.5)]
    FILLC = [(100.0, 150.0, 1.2)]
    # a lone cold stream reachable ONLY by H1, with a driving force barely above
    # DTMIN (tight LMTD): a capacity greedy typically never reaches because it
    # already spent H1's duty on the tempting C1 match.
    TIGHT1C = [(129.0, 189.0, 0.5)]
    # a lone cold stream reachable ONLY by the T2 group's hottest stream, same idea.
    TIGHT2C = [(199.0, 239.0, 0.5)]

    if tid == 1:
        H, C = [(200.0, 150.0, 2.0)], [(50.0, 120.0, 1.0)]
    elif tid == 2:
        H, C = [(200.0, 150.0, 2.0)], [(50.0, 120.0, 1.0), (130.0, 170.0, 1.0)]
    elif tid == 3:
        H, C = [(200.0, 150.0, 2.0), (90.0, 60.0, 1.0)], [(50.0, 120.0, 1.0)]
    elif tid == 4:
        H, C = list(T1H), list(T1C)
    elif tid == 5:
        H, C = list(T1H) + FILLH, list(T1C) + FILLC
    elif tid == 6:
        H, C = list(T1H) + FILLH, list(T1C) + FILLC + TIGHT1C
    elif tid == 7:
        H, C = list(T1H) + list(T2H), list(T1C) + list(T2C)
    elif tid == 8:
        H, C = list(T1H) + list(T2H) + FILLH, list(T1C) + list(T2C) + FILLC
    elif tid == 9:
        H, C = list(T1H) + list(T2H) + FILLH, list(T1C) + list(T2C) + FILLC + TIGHT1C
    else:  # tid == 10 (or beyond): largest / most combinatorial
        H, C = (list(T1H) + list(T2H) + FILLH,
                 list(T1C) + list(T2C) + FILLC + TIGHT1C + TIGHT2C)

    return H, C, dtmin, cH, cC, a


def main():
    tid = int(sys.argv[1])
    H, C, dtmin, cH, cC, a = instance(tid)
    out = [f"{len(H)} {len(C)}", f"{dtmin:.3f} {cH:.3f} {cC:.3f} {a:.3f}"]
    for (ths, tht, cp) in H:
        out.append(f"{ths:.3f} {tht:.3f} {cp:.3f}")
    for (tcs, tct, cp) in C:
        out.append(f"{tcs:.3f} {tct:.3f} {cp:.3f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
