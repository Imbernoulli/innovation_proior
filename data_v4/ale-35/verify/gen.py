#!/usr/bin/env python3
"""Instance generator for "Flood-Control Levee Placement" (grid flood / min-cut
flavoured optimisation, ALE-Bench).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    H W B S
    h_{0,0} h_{0,1} ... h_{0,W-1}
    ...
    h_{H-1,0} ... h_{H-1,W-1}
    sr_0 sc_0
    sr_1 sc_1
    ...
    sr_{S-1} sc_{S-1}

`H` rows and `W` cols of an integer HEIGHT grid; `B` the levee budget (number of
unit levees we may build); `S` the number of flood SOURCE cells, each given as a
(row, col) pair.

Flood semantics (see context.md): water starts at the source cells and spreads
to a 4-adjacent neighbour `v` from an already-flooded cell `u` whenever
`h[u] >= h[v]` -- i.e. water flows downhill or across level ground, never strictly
uphill. A levee placed on a cell makes that cell impassable: it never floods and
water cannot flow through it. Source cells are always flooded and can never carry
a levee.

The terrain is built so the levee placement is a genuine BOTTLENECK (min-cut)
problem, NOT a "wall the source" problem:

  * A wide flat RESERVOIR plain at a medium height fills the source region. The
    sources sit inside it, so the flood spreads across the WHOLE plain -- the
    plain is large, so sealing the source by ringing it is far too expensive for
    the budget B.
  * The plain is enclosed by tall RIDGES. Each ridge has only one or two low
    PASSES (gaps) punched through it. Beyond every pass lies a large DOWNSTREAM
    BASIN whose floor is LOWER than the plain, so once water reaches a pass it
    pours down and floods the entire basin.
  * Therefore the decisive move is to drop a levee IN each narrow pass: one levee
    in a pass can save a whole downstream basin. That is the min-cut bottleneck
    the strong heuristic must find -- a single flood pass reveals which passes
    carry the most downstream flooding.

H, W, B, S are chosen deterministically from the seed.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x35F1_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    H = rng.randint(28, 40)
    W = rng.randint(28, 40)
    S = rng.randint(1, 3)

    PLAIN = 55     # reservoir plain height
    RIDGE = 95     # ridge (wall) height
    BASIN = 20     # downstream basin floor height (lower than the plain)

    # Start everything at the plain height; carve basins below ridges later.
    grid = [[PLAIN for _ in range(W)] for _ in range(H)]

    # Choose a "reservoir" box (the source plain) somewhere in the middle, and
    # surround the rest of the map with basins separated by ridges. To keep the
    # construction simple and the bottleneck structure clean, we cut the grid
    # into horizontal STRIPS: the plain strip in the middle, and basin strips
    # above and below, each separated from the plain by a ridge row with passes.

    # plain strip rows [p0, p1)
    plain_h = rng.randint(max(6, H // 4), max(7, H // 3))
    p0 = rng.randint(H // 4, H // 2 - 2)
    p1 = min(H, p0 + plain_h)

    passes = []  # (r, c) of carved passes -- the bottlenecks

    def build_ridge_and_basin(ridge_row, basin_rows, vertical_dir):
        """Put a ridge across `ridge_row`, punch 1-2 passes, and lower the basin
        strip `basin_rows` (a range of rows) to BASIN height."""
        if ridge_row < 0 or ridge_row >= H:
            return
        for c in range(W):
            grid[ridge_row][c] = RIDGE
        npass = rng.randint(1, 2)
        cols = list(range(2, W - 2))
        rng.shuffle(cols)
        for pc in cols[:npass]:
            gap = rng.randint(PLAIN - 8, PLAIN)  # pass <= plain so water enters
            grid[ridge_row][pc] = gap
            passes.append((ridge_row, pc))
        for r in basin_rows:
            if 0 <= r < H:
                for c in range(W):
                    # only lower cells that are still plain-height (don't flatten
                    # other ridges); add a touch of noise for texture
                    grid[r][c] = BASIN  # flat basin: once water enters a pass it fills the whole basin

    # upper basin: ridge just above the plain, basin above that
    up_ridge = p0 - 1
    build_ridge_and_basin(up_ridge, range(0, up_ridge), +1)
    # lower basin: ridge just below the plain, basin below that
    lo_ridge = p1
    build_ridge_and_basin(lo_ridge, range(lo_ridge + 1, H), -1)

    # left/right edges: optionally a vertical ridge cutting a side basin out of
    # the plain, to add a second axis of bottlenecks for some seeds.
    if rng.random() < 0.6 and (p1 - p0) >= 5:
        side_left = rng.random() < 0.5
        if side_left:
            rc = rng.randint(3, max(4, W // 4))
            for r in range(p0, p1):
                grid[r][rc] = RIDGE
            for _ in range(rng.randint(1, 2)):
                pr = rng.randint(p0 + 1, p1 - 2)
                grid[pr][rc] = rng.randint(PLAIN - 8, PLAIN)
                passes.append((pr, rc))
            for r in range(p0, p1):
                for c in range(0, rc):
                    grid[r][c] = BASIN  # flat basin: once water enters a pass it fills the whole basin
        else:
            rc = rng.randint(min(W - 5, 3 * W // 4), W - 4)
            for r in range(p0, p1):
                grid[r][rc] = RIDGE
            for _ in range(rng.randint(1, 2)):
                pr = rng.randint(p0 + 1, p1 - 2)
                grid[pr][rc] = rng.randint(PLAIN - 8, PLAIN)
                passes.append((pr, rc))
            for r in range(p0, p1):
                for c in range(rc + 1, W):
                    grid[r][c] = BASIN  # flat basin: once water enters a pass it fills the whole basin

    # The plain is kept perfectly FLAT at PLAIN height: water from any source in
    # the plain then floods the whole plain (all cells equal => downhill-or-level
    # rule lets it spread freely), reaches every pass (gap <= PLAIN), and pours
    # into the lower basins. (No plain noise -- a single uphill bump would dam the
    # plain into disconnected puddles and destroy the bottleneck structure.)

    # budget: a few MORE than the number of passes, so the solver must pick the
    # right passes (and can't just plug literally everything for free, nor wall
    # the wide plain). Clamp to a sane band.
    npasses = len(passes)
    B = max(4, min(16, npasses + rng.randint(1, 4)))

    # sources: place them inside the plain strip, spread out.
    plain_cells = [(r, c) for r in range(p0, p1) for c in range(W)
                   if grid[r][c] >= PLAIN]
    rng.shuffle(plain_cells)
    chosen = []
    for (r, c) in plain_cells:
        ok = all(abs(pr - r) + abs(pc - c) >= 3 for (pr, pc) in chosen)
        if ok:
            chosen.append((r, c))
        if len(chosen) == S:
            break
    while len(chosen) < S and plain_cells:
        chosen.append(plain_cells[len(chosen) % len(plain_cells)])

    out = [f"{H} {W} {B} {S}"]
    for r in range(H):
        out.append(" ".join(str(grid[r][c]) for c in range(W)))
    for (r, c) in chosen[:S]:
        out.append(f"{r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
