import sys, random

# ---------------------------------------------------------------------------
# firebreak-ignition-lattice  (format C, maximize protected fuel)
#   `python3 gen.py <testId>`  prints ONE instance to stdout.
#   Deterministic in testId only.
#
# Instance:
#   line 1:  N F K
#   next K lines:  "r c"   ignition points (an 8x8 lattice over the map)
#   next N lines:  N integers  -- fuel[r][c] (>=1)
#
# The arsonist strikes at ONE of the K lattice points; fire percolates through
# 4-connected fuel cells not cut by a firebreak.  We must pre-cut <=F firebreak
# cells to minimise the WORST-CASE burned compartment (== maximise protected
# fuel over the worst ignition).  Fuel is planted with asymmetric "belts" so the
# obvious uniform / hot-spot cuts leave one over-fuelled compartment.
# ---------------------------------------------------------------------------

G = 8  # lattice side -> K = 64 ignition points (fixed across the ladder)

# Which ladder rungs plant a strongly-clustered (trap) fuel field.
TRAP_IDS = {2, 3, 5, 7, 9, 10}


def ign_positions(N):
    xs = []
    for k in range(G):
        p = int((k + 0.5) * N / G)
        p = max(0, min(N - 1, p))
        xs.append(p)
    # de-dup while preserving order (guaranteed distinct for N>=24, G=8)
    out = []
    for p in xs:
        if p not in out:
            out.append(p)
    return out


def main():
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    rng = random.Random(90000 + 137 * t)

    N = 24 + 2 * (t - 1)          # 24 .. 42
    K = G * G
    F = 4 * N                     # budget: ~4 full walls

    rows = ign_positions(N)
    cols = ign_positions(N)
    # if the ladder ever shrank N, rows/cols could shorten; pad defensively
    while len(rows) < G:
        rows.append(rows[-1])
    while len(cols) < G:
        cols.append(cols[-1])

    fuel = [[1] * N for _ in range(N)]
    background = N * N  # every cell starts at fuel 1

    # ---- plant SMALL, DENSE fuel blocks ------------------------------------
    # Each block is a small square (little AREA) holding a large share of the
    # total FUEL.  A uniform / equal-area cut can't slice inside such a block,
    # so whichever compartment holds it inherits a huge worst-case burn.  A
    # fuel-quantile cut, by contrast, spends walls right where the fuel is.
    trap = t in TRAP_IDS
    nb = 2 + (t % 3)                       # 2..4 dense blocks
    s = max(3, N // 6)                     # block side (small area)
    # value so all blocks together roughly match the background fuel (heavy skew)
    per_block_cells = s * s
    vblk = max(6, (background * (2 if trap else 1)) // (nb * per_block_cells))

    if trap:
        # cram the blocks into a single corner region (top-left quarter)
        anchors = [(rng.randint(0, max(0, N // 2 - s)),
                    rng.randint(0, max(0, N // 2 - s))) for _ in range(nb)]
    else:
        corners = [(0, 0), (0, N - s), (N - s, 0), (N - s, N - s), (N // 2, N // 2)]
        rng.shuffle(corners)
        anchors = []
        for k in range(nb):
            br, bc = corners[k % len(corners)]
            jr = rng.randint(-1, 1); jc = rng.randint(-1, 1)
            anchors.append((br + jr, bc + jc))

    for (r0, c0) in anchors:
        r0 = max(0, min(N - s, r0))
        c0 = max(0, min(N - s, c0))
        v = vblk + rng.randint(0, vblk // 4)
        for r in range(r0, r0 + s):
            for c in range(c0, c0 + s):
                fuel[r][c] += v

    # ---- emit --------------------------------------------------------------
    out = []
    out.append("%d %d %d" % (N, F, K))
    for r in rows:
        for c in cols:
            out.append("%d %d" % (r, c))
    for r in range(N):
        out.append(" ".join(str(fuel[r][c]) for c in range(N)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
