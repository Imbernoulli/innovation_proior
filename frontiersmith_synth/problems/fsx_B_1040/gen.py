#!/usr/bin/env python3
"""
gen.py <testId> -- Locator-beacon site generator (format C, family
anchor-set-geometry-regimes). Prints ONE instance to stdout.

Deterministic: every random choice is seeded from testId alone (no wall-clock,
no OS entropy). Builds a grid site map with two plazas ("open" regimes)
connected by a bent, width-2 corridor ("narrow" regime), plus a perimeter
sub-region of plaza1 and a junction sub-region straddling both corridor
mouths -- five qualitatively different target-distribution scenarios in all.

Output format:
  line 1:              W H A K T
  next H lines:         the grid, W chars each ('.' open, '#' wall)
  then, for each of the K scenarios:
      one line:          <scenario name>
      then T lines:       r c        (an open target cell)
"""
import random
import sys

A_ANCHORS = 12
T_TARGETS = 7


def carve_rect(grid, r0, r1, c0, c1):
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            grid[r][c] = '.'


def build_map(test_id):
    rng = random.Random(2000 + test_id)
    W = 20 + test_id            # 21 .. 30
    H = 12 + (test_id // 2)     # 12 .. 17
    grid = [['#'] * W for _ in range(H)]

    wA = max(4, W // 5)
    wB = max(4, W // 5)
    plazaA = (2, H - 3, 1, wA)              # r0,r1,c0,c1
    plazaB = (2, H - 3, W - 2 - wB, W - 2)
    carve_rect(grid, *plazaA)
    carve_rect(grid, *plazaB)

    midA = (plazaA[0] + plazaA[1]) // 2
    rowMid = max(plazaA[0], midA - 2)
    rowBendB = min(plazaB[1] - 1, midA + 3)
    cBend = (plazaA[3] + plazaB[2]) // 2

    # bent, WIDTH-2 corridor: horizontal out of plazaA -> vertical bend band
    # -> horizontal into plazaB. Width 2 (not 1) so a target deep in a
    # straight stretch still has a *little* off-axis LOS tolerance near a
    # bend/mouth -- narrow enough that only near-axis anchors are visible
    # from most of its length (the collinearity trap), but not literally
    # unsolvable.
    carve_rect(grid, rowMid, rowMid + 1, plazaA[3], cBend + 1)
    r0v, r1v = min(rowMid, rowBendB), max(rowMid + 1, rowBendB + 1)
    carve_rect(grid, r0v, r1v, cBend, cBend + 1)
    carve_rect(grid, rowBendB, rowBendB + 1, cBend, plazaB[2])

    # a few clutter obstacles inside each plaza (texture only; connectivity
    # is untouched since clutter never touches the corridor or the rims)
    for _ in range(2 + test_id % 3):
        r = rng.randint(plazaA[0] + 1, plazaA[1] - 1)
        c = rng.randint(plazaA[2] + 1, plazaA[3] - 1)
        if plazaA[1] - plazaA[0] > 3 and plazaA[3] - plazaA[2] > 3:
            grid[r][c] = '#'
    for _ in range(2 + test_id % 3):
        r = rng.randint(plazaB[0] + 1, plazaB[1] - 1)
        c = rng.randint(plazaB[2] + 1, plazaB[3] - 1)
        if plazaB[1] - plazaB[0] > 3 and plazaB[3] - plazaB[2] > 3:
            grid[r][c] = '#'

    meta = dict(plazaA=plazaA, plazaB=plazaB, rowMid=rowMid, rowBendB=rowBendB, cBend=cBend)
    return grid, W, H, meta


def in_rect(r, c, rect):
    r0, r1, c0, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def open_cells(grid):
    H = len(grid); W = len(grid[0])
    return [(r, c) for r in range(H) for c in range(W) if grid[r][c] == '.']


def build_scenarios(grid, meta, test_id):
    pa, pb = meta['plazaA'], meta['plazaB']
    allopen = open_cells(grid)
    plazaA_cells = [(r, c) for (r, c) in allopen if in_rect(r, c, pa)]
    plazaB_cells = [(r, c) for (r, c) in allopen if in_rect(r, c, pb)]
    corridor_cells = [(r, c) for (r, c) in allopen
                       if not in_rect(r, c, pa) and not in_rect(r, c, pb)]
    perimeterA = [(r, c) for (r, c) in plazaA_cells
                  if r == pa[0] or r == pa[1] or c == pa[2] or c == pa[3]]
    mouthA = (meta['rowMid'], pa[3])
    mouthB = (meta['rowBendB'], pb[2])

    def cheb(p, q):
        return max(abs(p[0] - q[0]), abs(p[1] - q[1]))

    junction = [(r, c) for (r, c) in allopen if cheb((r, c), mouthA) <= 3 or cheb((r, c), mouthB) <= 3]

    def pick(cells, k, seed):
        cells = sorted(set(cells))
        rng = random.Random(seed)
        if len(cells) <= k:
            # pad deterministically by cycling if a pathologically small pool occurs
            out = list(cells)
            i = 0
            while len(out) < k and cells:
                out.append(cells[i % len(cells)])
                i += 1
            return out
        return sorted(rng.sample(cells, k))

    scenarios = [
        ("plaza1", pick(plazaA_cells, T_TARGETS, 10 * test_id + 1)),
        ("plaza2", pick(plazaB_cells, T_TARGETS, 10 * test_id + 2)),
        ("corridor", pick(corridor_cells, T_TARGETS, 10 * test_id + 3)),
        ("perimeter1", pick(perimeterA, T_TARGETS, 10 * test_id + 4)),
        ("junction", pick(junction, T_TARGETS, 10 * test_id + 5)),
    ]
    return scenarios


def main():
    test_id = int(sys.argv[1])
    grid, W, H, meta = build_map(test_id)
    scenarios = build_scenarios(grid, meta, test_id)

    out = []
    out.append(f"{W} {H} {A_ANCHORS} {len(scenarios)} {T_TARGETS}")
    for row in grid:
        out.append(''.join(row))
    for name, targets in scenarios:
        out.append(name)
        for (r, c) in targets:
            out.append(f"{r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
