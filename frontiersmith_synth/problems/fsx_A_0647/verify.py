import sys

TAU = 2               # aTAM temperature
MAX_M = 5000           # sanity cap on declared tile-type count
MAX_LABEL = 10 ** 6    # sanity cap on glue-label integers
MARGIN = 8              # scan margin beyond the target bounding box


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp_toks = open(sys.argv[1]).read().split()
        T = int(inp_toks[0])
    except Exception:
        fail("bad input")
    if T < 0:
        fail("bad input T<0")

    k = max(1, T.bit_length())
    # k==1: a single column can locally detect "just decremented past zero" and
    # halt on its own -> height T+1 rows suffice. k>=2 needs a "relay" row
    # inserted after every counting row so the leftmost column only advances
    # once it hears back that the rightmost (most-significant) column actually
    # continued -> height doubles (2T+1 rows: T+1 counting rows + T relay rows).
    height = (T + 1) if k == 1 else (2 * T + 1)
    target = set((x, y) for x in range(k) for y in range(height))
    if k > 1:
        # the topmost relay attempt at every column except the least-
        # significant one is a deterministic, unavoidable "flag" row left by
        # any construction that halts a temperature-2 counter via a
        # relay-style global zero check: only the least-significant column
        # can locally refuse to continue, so every OTHER column still
        # legitimately completes its one final relay step before the chain
        # as a whole comes up short.
        target.update((c, height) for c in range(1, k))
    B = len(target)  # internal baseline: one unique tile type per cell ("the map")

    # ---------------- parse participant output ----------------
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    it = iter(raw)

    def next_int():
        return int(next(it))

    try:
        M = next_int()
    except Exception:
        fail("missing tile count")
    if M < 1 or M > MAX_M:
        fail("tile count %d out of range" % M)

    types = []
    try:
        for _ in range(M):
            row = [next_int() for _ in range(8)]
            for v in row:
                if v < 0 or v > MAX_LABEL:
                    fail("label/strength out of range")
            Nl, Ns, El, Es, Sl, Ss, Wl, Ws = row
            for (lab, st) in ((Nl, Ns), (El, Es), (Sl, Ss), (Wl, Ws)):
                if st not in (0, 1, 2):
                    fail("strength must be 0/1/2")
                if (lab == 0) != (st == 0):
                    fail("null glue (label=0) must have strength=0 and vice versa")
            types.append(row)
    except (StopIteration, ValueError):
        fail("malformed tile type row")

    try:
        seed_idx = next_int()
    except Exception:
        fail("missing seed index")
    if seed_idx < 1 or seed_idx > M:
        fail("seed index out of range")

    # no trailing garbage tokens allowed (bounded-read discipline)
    leftover = list(it)
    if leftover:
        fail("trailing garbage after declared fields")

    # ---------------- deterministic aTAM simulation ----------------
    # side order per type row: 0:Nl 1:Ns 2:El 3:Es 4:Sl 5:Ss 6:Wl 7:Ws
    grid = {}
    grid[(0, 0)] = seed_idx - 1

    xlo, xhi = -MARGIN, (k - 1) + MARGIN
    ylo, yhi = -MARGIN, (height - 1) + MARGIN
    area = B
    MAXSTEPS = 20 * area + 2000

    steps = 0

    def side_of(tp, direction):
        # returns (label, strength) for direction in 'N','E','S','W'
        idx = {"N": 0, "E": 2, "S": 4, "W": 6}[direction]
        return tp[idx], tp[idx + 1]

    OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
    DELTA = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}

    # Event-driven fixpoint: only (re-)examine a position when a neighbor of
    # it has just been filled, instead of repeatedly re-scanning the whole
    # bounding box. This is equivalent to the fixed-point of repeated full
    # sweeps (same deterministic ambiguity/placement rule) but O(area) total
    # work instead of O(area * passes) -- needed because a placement CHAIN
    # (e.g. the trivial one-tile-per-cell "map") can need up to O(area)
    # sweep-passes to fully propagate under a naive full rescan.
    from collections import deque
    frontier = deque()
    in_frontier = set()

    def consider(pos):
        x, y = pos
        if not (xlo <= x <= xhi and ylo <= y <= yhi):
            return
        if pos in grid or pos in in_frontier:
            return
        in_frontier.add(pos)
        frontier.append(pos)

    sx, sy = 0, 0
    for _d, (dx, dy) in DELTA.items():
        consider((sx + dx, sy + dy))

    while frontier:
        pos = frontier.popleft()
        in_frontier.discard(pos)
        if pos in grid:
            continue
        x, y = pos
        nbr = {}
        any_nbr = False
        for d, (dx, dy) in DELTA.items():
            npos = (x + dx, y + dy)
            if npos in grid:
                nbr[d] = types[grid[npos]]
                any_nbr = True
        if not any_nbr:
            continue
        candidates = []
        for ti, tp in enumerate(types):
            total = 0
            for d in ("N", "E", "S", "W"):
                if d not in nbr:
                    continue
                my_lab, my_st = side_of(tp, d)
                if my_lab == 0:
                    continue
                their_lab, their_st = side_of(nbr[d], OPP[d])
                if their_lab == my_lab and their_st == my_st:
                    total += my_st
            if total >= TAU:
                candidates.append(ti)
        if len(candidates) > 1:
            fail("ambiguous assembly at (%d,%d): %d candidate tile types bind simultaneously" % (x, y, len(candidates)))
        if len(candidates) == 1:
            grid[pos] = candidates[0]
            steps += 1
            if steps > MAXSTEPS:
                fail("assembly did not terminate (runaway growth, > %d placements)" % MAXSTEPS)
            for d, (dx, dy) in DELTA.items():
                consider((x + dx, y + dy))

    filled = set(grid.keys())
    if filled != target:
        missing = len(target - filled)
        extra = len(filled - target)
        fail("terminal assembly != target shape (missing=%d extra=%d)" % (missing, extra))

    F = M
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("T=%d k=%d h=%d B=%d F=%d Ratio: %.6f" % (T, k, height, B, F, sc / 1000.0))


if __name__ == "__main__":
    main()
