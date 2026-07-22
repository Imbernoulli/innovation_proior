#!/usr/bin/env python3
"""Generator for fsx_A_0947 -- Rising Water, Staged Barriers.
Usage: gen.py <testId>   (1..10). Deterministic given testId. Prints one test to stdout.

Terrain layout (all deterministic from testId via a seeded RNG):
  - R x C grid. Border ring (r==0/R-1 or c==0/C-1) is elevation 0 -- the permanent
    "sea"/source. Interior default cells are shallow "plain" elevation (sea-like).
  - NB+1 rectangular basins are placed in a single row: NB are PROTECTED zones (P),
    the last is the FORBIDDEN zone (F). Each basin has a 3x3 low-elevation interior
    fully surrounded by a 2-cell-thick MOUNTAIN moat (never floods), except for a
    narrow "gate" -- gate_width parallel 2-deep tunnels of moderate elevation carved
    through the moat's north side, which is the *only* route water can ever use to
    reach that basin's interior.
  - On some (trap) tests, basin P_0 additionally gets a "decoy" notch: a few cells on
    the WEST outer moat layer are lowered to a (low, early-activating) pass elevation,
    but the moat's inner layer behind them is left as solid mountain, so the notch is
    a dead-end pocket close to the protected interior by raw distance yet topologically
    useless -- it never connects to the basin at all.
"""
import sys

SEA_LO, SEA_HI = 1, 4
BASIN_LOW = 2
MOUNTAIN = 900
DECOY_ELEV = 16          # its own stage, strictly between the sea stage and any gate
GATE_BASE = 30
GATE_STEP = 40
DECOY_WIDTH = 3
BW = 10                 # column band width per basin


class RNG:
    """Tiny deterministic LCG so we never depend on Python's random module version."""
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def randint(self, lo, hi):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return lo + self.s % (hi - lo + 1)


def test_params(t):
    nb_table = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4}
    k_table = {1: 4, 2: 4, 3: 5, 4: 5, 5: 5, 6: 6, 7: 6, 8: 6, 9: 7, 10: 7}
    trap_set = {2, 4, 6, 8, 10}
    NB = nb_table[t]
    K = k_table[t]
    trap = t in trap_set
    return NB, K, trap


def build(t):
    NB, K, trap = test_params(t)
    rnd = RNG(1000 + t * 7919)

    total_basins = NB + 1  # NB protected + 1 forbidden
    C = 3 + total_basins * BW + 3
    R = 16
    ir0, ir1 = 6, 8  # interior rows for every basin (3 rows)

    elev = [[rnd.randint(SEA_LO, SEA_HI) for _ in range(C)] for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if r == 0 or r == R - 1 or c == 0 or c == C - 1:
                elev[r][c] = 0

    gate_widths = []
    pass_elevs = []
    basin_cols = []
    for i in range(total_basins):
        ic0 = 5 + i * BW
        basin_cols.append(ic0)
        gw = 1 + (i % 3)
        if i == total_basins - 1:  # F basin: keep its gate cheap & uniform
            gw = 1
        gate_widths.append(gw)
        pass_elevs.append(GATE_BASE + i * GATE_STEP)

    p_groups = []  # list of list[(r,c)] for each protected basin
    f_cells = []

    for i in range(total_basins):
        ic0 = basin_cols[i]
        ic1 = ic0 + 2
        gw = gate_widths[i]
        pe = pass_elevs[i]
        # 2-thick moat frame -> MOUNTAIN
        for r in range(ir0 - 2, ir1 + 3):
            for c in range(ic0 - 2, ic1 + 3):
                if ir0 <= r <= ir1 and ic0 <= c <= ic1:
                    continue
                elev[r][c] = MOUNTAIN
        # interior -> low basin floor
        for r in range(ir0, ir1 + 1):
            for c in range(ic0, ic1 + 1):
                elev[r][c] = BASIN_LOW
        # gate: north side, 2 layers deep, gw columns wide, elevation pe
        for r in (ir0 - 2, ir0 - 1):
            for c in range(ic0, ic0 + gw):
                elev[r][c] = pe
        # decoy: only on the FIRST protected basin, only on trap tests -- west side,
        # OUTER layer only (inner layer stays MOUNTAIN => dead end, never reaches P)
        if trap and i == 0:
            for r in range(ir0, ir0 + DECOY_WIDTH):
                elev[r][ic0 - 2] = DECOY_ELEV

        cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
        if i < total_basins - 1:
            p_groups.append(cells)
        else:
            f_cells = cells

    total_cut_cost = sum(gate_widths)
    # F's own gate is always the cheapest possible (width 1); a naive solver that
    # ignores the forbidden zone entirely never contests this budget, so the trap
    # only needs to out-cost the *protected*-only budget, not the combined total.
    slack = 1 if trap else max(4, total_cut_cost)
    total_budget = total_cut_cost + slack

    # ---- level schedule: one stage boundary per activation threshold, in order,
    # so each threshold is reached (and hence detectable-vs-hidden) on its own
    # stage -- this is what makes the decoy's early, throwaway activation and the
    # real gates' later activations land on genuinely DIFFERENT decision rounds.
    thresholds = []
    if trap:
        thresholds.append(DECOY_ELEV)
    thresholds.extend(pass_elevs)
    thresholds.sort()

    levels = [SEA_HI + 2]
    prev = levels[0]
    for th in thresholds:
        v = max(th, prev + 1)
        levels.append(v)
        prev = v
    while len(levels) < K:
        levels.append(prev + 5)
        prev = levels[-1]
    levels = levels[:K]

    if trap:
        # The decoy is always the smallest threshold, i.e. it activates at stage
        # index 1 (levels[1]). Guarantee the *full* decoy_width is affordable by
        # then (else a per-stage cap could truncate the waste and accidentally
        # save the real gate) by front-loading exactly DECOY_WIDTH units across
        # stages 1..2, then spreading whatever budget remains across the rest.
        w0 = DECOY_WIDTH - 1
        w1 = 1
        remaining = total_budget - DECOY_WIDTH
        rest_stages = max(1, K - 2)
        base = remaining // rest_stages
        rem = remaining - base * rest_stages
        W = [w0, w1] + [base] * (K - 2)
        for i in range(rem):
            W[2 + (i % max(1, K - 2))] += 1
    else:
        base_w = total_budget // K
        rem = total_budget - base_w * K
        W = [base_w] * K
        for i in range(rem):
            W[i % K] += 1
        if W[0] < 1:
            W[0] = 1  # never start with a literally-zero budget stage

    alpha = 0.35 + 0.02 * (t % 5)  # in [0.35, 0.43], varies per test, read from input

    out = []
    out.append(f"{R} {C} {K}")
    for row in elev:
        out.append(" ".join(map(str, row)))
    out.append(str(NB))
    for grp in p_groups:
        out.append(str(len(grp)))
        for (r, c) in grp:
            out.append(f"{r} {c}")
    out.append(str(len(f_cells)))
    for (r, c) in f_cells:
        out.append(f"{r} {c}")
    out.append(" ".join(map(str, levels)))
    out.append(" ".join(map(str, W)))
    out.append(f"{alpha:.4f}")
    sys.stdout.write("\n".join(out) + "\n")


def main():
    t = int(sys.argv[1])
    build(t)


if __name__ == "__main__":
    main()
