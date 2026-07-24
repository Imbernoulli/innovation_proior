#!/usr/bin/env python3
"""gen.py <testId> -- generator for 'Ashline Refinery Bootstrap' (fsx_A_1102).

10-case difficulty ladder. All randomness is seeded from testId only.
Prints ONE instance to stdout:

  T K M
  F0 C0 Cmax cu beta v
  P kappa
  d_1 S_1
  ...
  d_M S_M
"""
import sys
import random


def emit(T, K, M, F0, C0, Cmax, cu, beta, v, P, kap, mines):
    lines = []
    lines.append("%d %d %d" % (T, K, M))
    lines.append("%.6f %.6f %.6f %.6f %.6f %.6f" % (F0, C0, Cmax, cu, beta, v))
    lines.append("%.6f %.6f" % (P, kap))
    for d, s in mines:
        lines.append("%.6f %.6f" % (d, s))
    return "\n".join(lines) + "\n"


def jitter_mines(rnd, spec):
    """spec: list of (distance, stock). Apply small deterministic jitter."""
    out = []
    for d, s in spec:
        dj = d * (1.0 + rnd.uniform(-0.06, 0.06))
        sj = s * (1.0 + rnd.uniform(-0.10, 0.10))
        out.append((round(dj, 3), round(sj, 3)))
    return out


def main():
    tid = int(sys.argv[1])
    assert 1 <= tid <= 10
    rnd = random.Random(1000003 * tid + 77)
    v = 40.0
    P = 10.0
    kap = 0.30

    if tid == 1:
        # small, benign: upgrades modestly useful, all mines profitable
        mines = jitter_mines(rnd, [(25, 150), (35, 220), (55, 300), (80, 420)])
        print(emit(120, 3, 4, 120.0, 5.0, 18.0, 90.0, 8.0, v, P, kap, mines), end="")
    elif tid == 2:
        # small-moderate: two near mines, one mid
        mines = jitter_mines(rnd, [(22, 130), (30, 200), (48, 260), (65, 340),
                                   (90, 420)])
        print(emit(160, 4, 5, 110.0, 4.5, 22.0, 70.0, 8.0, v, P, kap, mines), end="")
    elif tid == 3:
        # TRAP bootstrap-crunch: tiny F0, tiny C0, cheap capacity, long horizon.
        # Compounding dominates; per-trip efficiency plateaus.
        mines = jitter_mines(rnd, [(18, 90), (26, 160), (40, 260), (58, 340),
                                   (75, 420), (95, 520)])
        print(emit(220, 5, 6, 60.0, 3.0, 40.0, 35.0, 9.0, v, P, kap, mines), end="")
    elif tid == 4:
        # TRAP bootstrap-crunch, harder: even poorer start, richer far mines
        mines = jitter_mines(rnd, [(16, 80), (24, 150), (38, 240), (52, 320),
                                   (70, 430), (88, 560), (105, 640), (120, 720)])
        print(emit(260, 6, 8, 50.0, 2.5, 48.0, 30.0, 10.0, v, P, kap, mines), end="")
    elif tid == 5:
        # cheap upgrades + very long horizon: growth phase must run long
        mines = jitter_mines(rnd, [(20, 120), (28, 180), (42, 260), (60, 360),
                                   (78, 460), (95, 560), (110, 640), (125, 720),
                                   (140, 780), (150, 840)])
        print(emit(300, 6, 10, 80.0, 3.5, 60.0, 26.0, 8.0, v, P, kap, mines), end="")
    elif tid == 6:
        # TRAP concavity: ore plentiful and near; full-throttle refining wastes
        # ~half the yield. Spreading the run level is the win; upgrades meh.
        mines = jitter_mines(rnd, [(15, 600), (22, 800), (30, 900), (40, 1100),
                                   (55, 900), (70, 700)])
        print(emit(200, 5, 6, 150.0, 6.0, 24.0, 95.0, 10.0, v, P, kap, mines), end="")
    elif tid == 7:
        # concavity + moderate upgrade value mixed
        mines = jitter_mines(rnd, [(18, 400), (26, 500), (36, 600), (50, 650),
                                   (68, 600), (85, 520), (100, 460)])
        print(emit(240, 5, 7, 120.0, 5.0, 30.0, 60.0, 9.0, v, P, kap, mines), end="")
    elif tid == 8:
        # TRAP net-negative far mines: greedy keeps hauling ore whose fuel cost
        # exceeds the fuel it refines into; strong refuses.
        mines = jitter_mines(rnd, [(20, 140), (30, 220), (45, 260),
                                   (130, 520), (160, 820), (190, 1200)])
        print(emit(200, 5, 6, 130.0, 5.0, 20.0, 80.0, 7.0, v, P, kap, mines), end="")
    elif tid == 9:
        # mixed: tight start + tempting far big-stock mines + cheap upgrades
        mines = jitter_mines(rnd, [(17, 90), (25, 160), (40, 240), (60, 340),
                                   (85, 480), (120, 700), (150, 900), (175, 1100),
                                   (195, 1300)])
        print(emit(280, 6, 9, 55.0, 3.0, 44.0, 32.0, 8.5, v, P, kap, mines), end="")
    else:
        # large adversarial mix: everything at once, biggest instance
        spec = [(15, 100), (20, 140), (26, 180), (32, 220), (40, 260), (50, 320),
                (62, 380), (75, 440), (88, 520), (100, 560), (112, 620),
                (125, 700), (138, 760), (150, 820), (162, 880), (172, 940),
                (180, 1000), (188, 1060), (194, 1120), (198, 1180),
                (120, 500), (95, 420), (58, 300), (35, 240)]
        mines = jitter_mines(rnd, spec)
        print(emit(400, 16, 24, 70.0, 3.0, 50.0, 30.0, 9.0, v, P, kap, mines), end="")


if __name__ == "__main__":
    main()
