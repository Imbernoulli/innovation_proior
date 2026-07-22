#!/usr/bin/env python3
"""
gen.py <testId> -- prints ONE instance of the reaction-network firing-sequence
problem to stdout. Deterministic: seeded ONLY by testId.

Instance format:
  line 1: m N_R N_Z
  next m lines (one per group g = 0..m-1):
      px py dx dv cx cy cw

Species per group g: intermediates X_g, Y_g.
Reactions per group g:
  Px_g:    px_g R            -> 1 X_g
  Py_g:    py_g R            -> 1 Y_g
  Dir_g:   dx_g X_g          -> dv_g T
  Combo_g: cx_g X_g + cy_g Y_g + 1 Z -> cw_g T

R (raw material, initial N_R) and Z (catalyst, initial N_Z) are the two
irreversibly-consumed shared resources. Group 0 is a deliberately weak
"do-nothing-clever" baseline group. Group 1 has the best pure-direct rate
(the visible, locally-tempting route). Group 2 has a MUCH better combo
rate than group 1's direct rate, but needs the scarce catalyst Z plus a
reserved stock of Y_g (which is USELESS on its own) -- the trap.
"""
import sys


def main():
    testId = int(sys.argv[1])
    import random
    rng = random.Random(900701 + 37 * testId)

    # ladder sizing: more groups + bigger budgets as testId grows
    m = 3 + min(3, (testId - 1) // 3)          # 3,3,3,4,4,4,5,5,5,6
    N_R = 20 * testId + rng.randint(0, 12)     # grows to ~210+ at testId=10

    groups = []

    # ---- group 0: weak baseline (mediocre at everything) ----
    px0, py0 = rng.randint(2, 4), rng.randint(2, 4)
    dx0 = rng.randint(2, 4)
    rate0 = rng.uniform(0.8, 1.2)
    dv0 = max(1, round(rate0 * dx0 * px0))
    cx0, cy0 = rng.randint(1, 3), rng.randint(1, 3)
    crate0 = rng.uniform(0.8, 1.2)
    cw0 = max(1, round(crate0 * (cx0 * px0 + cy0 * py0)))
    groups.append([px0, py0, dx0, dv0, cx0, cy0, cw0])

    # ---- group 1: direct-favored decoy (best VISIBLE per-step rate) ----
    px1, py1 = rng.randint(1, 3), rng.randint(1, 3)
    dx1 = rng.randint(1, 2)
    rate1 = rng.uniform(2.4, 3.2)
    dv1 = max(1, round(rate1 * dx1 * px1))
    cx1, cy1 = rng.randint(1, 3), rng.randint(1, 3)
    crate1 = rng.uniform(1.0, 1.6)
    cw1 = max(1, round(crate1 * (cx1 * px1 + cy1 * py1)))
    groups.append([px1, py1, dx1, dv1, cx1, cy1, cw1])

    # ---- group 2: combo-favored true optimum ----
    # combo rate is FORCED to beat group1's direct rate by a healthy,
    # testId-independent margin so the trap never washes out at scale.
    px2, py2 = rng.randint(1, 2), rng.randint(1, 2)
    dx2 = rng.randint(2, 4)
    rate2 = rng.uniform(1.1, 1.7)                  # its OWN direct route: mediocre
    dv2 = max(1, round(rate2 * dx2 * px2))
    cx2, cy2 = rng.randint(1, 2), rng.randint(1, 2)
    cost2 = cx2 * px2 + cy2 * py2
    crate2 = rate1 + rng.uniform(1.5, 2.5)          # combo route: big premium over rate1
    cw2 = max(1, round(crate2 * cost2))
    groups.append([px2, py2, dx2, dv2, cx2, cy2, cw2])

    # ---- filler / noise groups (kept strictly worse than group1/group2) ----
    for gi in range(3, m):
        pxf, pyf = rng.randint(1, 4), rng.randint(1, 4)
        dxf = rng.randint(1, 4)
        ratef = rng.uniform(0.8, 2.2)               # < group1's min direct rate (2.4)
        dvf = max(1, round(ratef * dxf * pxf))
        cxf, cyf = rng.randint(1, 3), rng.randint(1, 3)
        cratef = rng.uniform(0.8, 3.0)               # < group2's min combo rate
        cwf = max(1, round(cratef * (cxf * pxf + cyf * pyf)))
        groups.append([pxf, pyf, dxf, dvf, cxf, cyf, cwf])

    # N_Z sized so the catalyst can absorb a SUBSTANTIAL (but still scarce
    # -- integer-bounded) share of the raw budget if routed entirely through
    # group 2's combo: N_Z * cost2 ~= 0.6 * N_R.
    N_Z = max(3, round(0.6 * N_R / max(1, cost2)))
    N_Z = min(N_Z, 60)

    print(m, N_R, N_Z)
    for g in groups:
        print(*g)


if __name__ == "__main__":
    main()
