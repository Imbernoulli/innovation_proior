#!/usr/bin/env python3
"""gen.py <testId> -> prints one instance of the reinsurance-layer-buy problem to
stdout. Deterministic: all randomness is seeded from testId only.

Catalog: 6 excess-of-loss layers forming a contiguous tower above the cedant's
retention. Rate-on-line (premium / width) is strictly CHEAPEST at the top of the
tower and most expensive for the layer sitting right above retention -- so a
naive "buy the cheapest rate-on-line first" shopper always leaves the bottom of
the tower (near retention) as the last thing funded. Scenario sets: on ordinary
years (most test ids) loss severity is concentrated where that cheap coverage
actually sits (mid/upper tower), so the cheap-first shopper does fine. On the
three TRAP test ids, whole scenarios are engineered so most losses land squarely
in the band the cheap-first shopper leaves open -- exposing the hole.
"""
import sys
import random

A0_LIST = [8, 9, 10, 10, 11, 13, 14, 16, 18, 19]
BW_LIST = [10, 11, 12, 13, 14, 16, 18, 20, 22, 24]
PSCALE = [1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
PREM_BASE = [10, 8, 6, 5, 4, 3]     # premium for 100% placement, band0..band5
REINST_K = [1, 1, 2, 2, 3, 3]        # extra reinstatements per band
REINST_PCT = [35, 30, 20, 15, 10, 5]  # reinstatement cost, % of recovery
TRAP_IDS = {3, 6, 9}
NSCEN = 40
TRAP_N1, TRAP_N2 = 8, 2
TRAP_LO, TRAP_HI = 1.3, 1.97
PFRAC = 0.38


def build(test_id):
    A0 = A0_LIST[test_id - 1]
    BW = BW_LIST[test_id - 1]
    sc = PSCALE[test_id - 1]

    catalog = []
    A = A0
    for i in range(6):
        P = max(1, round(PREM_BASE[i] * sc))
        catalog.append((A, BW, P, REINST_K[i], REINST_PCT[i]))
        A += BW

    total_prem = sum(c[2] for c in catalog)
    C0 = round(108.9 * sc)
    Pmax = int(total_prem * PFRAC)

    is_trap = test_id in TRAP_IDS
    rng = random.Random(2000 + test_id * 131)
    scens = []
    for s in range(NSCEN):
        evs = []
        if is_trap and s % 4 != 3:
            if s % 4 == 0:
                # attritional cluster squarely in the band the cheap-first
                # shopper leaves open (just above retention)
                for _ in range(TRAP_N1):
                    evs.append(rng.uniform(A0 + TRAP_LO * BW, A0 + TRAP_HI * BW))
            else:
                for _ in range(TRAP_N2):
                    evs.append(rng.uniform(A0 + TRAP_LO * BW, A0 + TRAP_HI * BW))
                evs.append(rng.uniform(A0 + 0.1 * BW, A0 + 1.0 * BW))
        else:
            # ordinary year: severity concentrated where the cheap layers sit
            for _ in range(rng.randint(1, 3)):
                if rng.random() < 0.85:
                    evs.append(rng.uniform(A0 + 2.0 * BW, A0 + 6.0 * BW))
                else:
                    evs.append(rng.uniform(A0 * 0.3, A0 + 2.0 * BW))
        evs = [max(1, round(x)) for x in evs]
        rng.shuffle(evs)
        scens.append(evs)

    return catalog, C0, Pmax, scens


def main():
    test_id = int(sys.argv[1])
    catalog, C0, Pmax, scens = build(test_id)
    lines = [str(len(catalog))]
    for (A, W, Prem, K, RP) in catalog:
        lines.append(f"{A} {W} {Prem} {K} {RP}")
    lines.append(f"{C0} {Pmax}")
    lines.append(str(len(scens)))
    for evs in scens:
        lines.append(f"{len(evs)} " + " ".join(map(str, evs)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
