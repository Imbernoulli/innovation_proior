#!/usr/bin/env python3
"""gen.py <testId> -> prints one roof-drainage instance to stdout.

Deterministic: all randomness seeded ONLY from testId.

Construction:
  - Base terrain = a "hip roof": height(r,c) = dist-to-nearest-edge (integer),
    so it already drains smoothly to the border everywhere it is undisturbed.
    Every border cell is a gutter.
  - We carve NPITS rectangular "cisterns" (pits) into the interior. Each pit's
    interior is a shallow bowl sloping toward one boundary cell (the "sill").
    Immediately outside the pit we raise a ring of wall cells (a temple-roof
    parapet) so that, together with the max-slope relaxation pass, the pit is
    fully enclosed: no droplet dropped anywhere in the pit reaches a gutter.
  - On exactly one side of each pit, the wall is calibrated (via the natural
    height two rings further out, "beyond") so that a SINGLE cell -- the
    "gate" -- can be lowered by a small, constant edit to reconnect the whole
    pit to the already-draining exterior. Every other ring cell is a much
    more expensive breach. This is the ridgeline: the score changes only when
    an edit flips a ridge cell, not when it moves surface elsewhere.
  - For a MINORITY of pits (probability ALIGN_PROB) the gate sits on the side
    nearest the border -- the direction a naive "straight channel to the
    nearest gutter" recipe would also aim at, so that recipe occasionally
    works and is not a total dead end. For the rest, the gate is calibrated
    on a different (second-nearest) side: the naive straight line then runs
    into solid, unbroken parapet on the nearest side instead and is blocked
    outright, while the ridge-only search (which looks at the SINK's own
    neighbors, not "nearest border") still finds the true gate immediately.
  - A handful of single obstacle cells (fixed pillars) are scattered in the
    open terrain; they are impassable (excluded from the flow graph) and
    cannot be edited, and incidentally create a few small extra pockets.
"""
import sys, random

B_MULT = 2.35       # edit budget = B_MULT * sum of real (confirmed-working) gate costs
ALIGN_PROB = 0.35   # fraction of pits whose gate sits on the naive-nearest side

DIRS = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # Up, Left, Down, Right -- fixed tie-break order
BIG = 10 ** 6


def dist_to_edge(r, c, R, C):
    return min(r, R - 1 - r, c, C - 1 - c)


def relax(H, obstacle, R, C, S):
    """Repeatedly clip H[a] down to H[b]+S for every non-obstacle adjacent pair,
    until a fixpoint. Only lowers values -> always terminates, keeps everything
    an S-Lipschitz (slope-feasible) heightfield."""
    changed = True
    guard = 0
    while changed and guard < 4000:
        changed = False
        guard += 1
        for r in range(R):
            for c in range(C):
                if obstacle[r][c]:
                    continue
                base = H[r][c]
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C and not obstacle[nr][nc]:
                        cap = H[nr][nc] + S
                        if base > cap:
                            base = cap
                if base != H[r][c]:
                    H[r][c] = base
                    changed = True
    return guard


def route(H, obstacle, gutter, R, C, start, maxsteps):
    pos = start
    for _ in range(maxsteps):
        r, c = pos
        if gutter[r][c]:
            return True
        best = None
        bh = None
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not obstacle[nr][nc] and H[nr][nc] < H[r][c]:
                if bh is None or H[nr][nc] < bh:
                    bh = H[nr][nc]
                    best = (nr, nc)
        if best is None:
            return False
        pos = best
    return False


def place_pits(R, C, npits, pit_h, pit_w, margin_border, rng):
    """Lattice-based placement of non-overlapping pits, each needing a
    1-cell ring buffer plus margin_border clearance from the grid border."""
    lo_r, hi_r = margin_border, R - margin_border - pit_h
    lo_c, hi_c = margin_border, C - margin_border - pit_w
    if hi_r < lo_r or hi_c < lo_c:
        return []
    cell_h = pit_h
    cell_w = pit_w
    slots = []
    r = lo_r
    while r <= hi_r:
        c = lo_c
        while c <= hi_c:
            slots.append((r, c))
            c += cell_w
        r += cell_h
    rng.shuffle(slots)
    return slots[:npits]


def build(testId):
    rng = random.Random(20000 + 97 * testId)

    # ---- difficulty ladder ----
    R = 70 + 24 * testId
    C = 82 + 24 * testId
    S = 5
    MARGIN = 3                     # sill sits MARGIN above the natural terrain 2 rings beyond the gate
    MAX_PIT_H, MAX_PIT_W = 30, 35
    SIZES = [(26, 29), (28, 31), (30, 33), (30, 35)]  # varied basin sizes
                                    # a naive "biggest puddle first" recipe will chase the huge
                                    # ones even when they are far from any gutter, instead of the
                                    # many small ones that are just as cheap to breach as the big
    npits = 9999                   # pack the lattice as densely as it allows
    nobstacles = 2                 # a couple of decorative fixed pillars

    H = [[dist_to_edge(r, c, R, C) for c in range(C)] for r in range(R)]
    obstacle = [[False] * C for _ in range(R)]
    gutter = [[dist_to_edge(r, c, R, C) == 0 for c in range(C)] for r in range(R)]

    # No border margin: pits pack right up against the lattice edge so the
    # interior is used as densely as possible. The trap does not come from
    # distance-to-border (the ridge-gate cost is flat regardless of it) --
    # it comes from which SIDE of the pit the gate is on (see ALIGN_PROB
    # above).
    margin_border = 0
    slots = place_pits(R, C, npits, MAX_PIT_H, MAX_PIT_W, margin_border=margin_border, rng=rng)

    ring_cells = set()
    pit_cells = set()
    reserved = set()   # pit + wall + the gate's escape chain: keep obstacles out
    gates = []
    for (r0, c0) in slots:
        pit_h, pit_w = SIZES[rng.randrange(len(SIZES))]
        r1, c1 = r0 + pit_h, c0 + pit_w
        pit_rect = {(r, c) for r in range(r0, r1) for c in range(c0, c1)}
        # a pit could pick ANY side as its gate, so its whole 1-ring wall must be
        # clear of every previously-placed pit's reserved footprint, or the two
        # would corrupt each other's calibration.
        touch_ring = set()
        for (r, c) in pit_rect:
            for ddr, ddc in DIRS:
                nr, nc = r + ddr, c + ddc
                touch_ring.add((nr, nc))
        if any((r, c) in reserved for (r, c) in touch_ring):
            continue
        # Candidate drain sides, one per wall: each candidate's "beyond" cell
        # (2 rings past the wall) sits at some natural distance-to-border.
        # The naive "straight line to the NEAREST gutter" that a first-idea
        # solver would carve always aims at the CLOSEST side (rank 0 below).
        # For most pits we deliberately calibrate the actual ridge gate on a
        # *different* side (rank != 0): that side's wall still has exactly
        # one working hole (the gate), but it is NOT the side a
        # nearest-gutter straight line would ever reach, so that naive line
        # runs straight into solid, unbroken parapet stonework on the
        # (still-intact) nearest side and is blocked outright, while the
        # ridge-only reconnect (which searches the sink's own neighbors, not
        # "nearest border") finds the true, still-cheap gate regardless of
        # which side it is on. A minority of pits keep the aligned (rank 0)
        # gate -- the recipe is not USELESS, it recovers the easy pits, it
        # just cannot find the majority that need the insight.
        cands = []
        for gdir in range(4):
            dr, dc = DIRS[gdir]
            if dr == -1: s = (r0, (c0 + c1 - 1) // 2)
            elif dr == 1: s = (r1 - 1, (c0 + c1 - 1) // 2)
            elif dc == -1: s = ((r0 + r1 - 1) // 2, c0)
            else: s = ((r0 + r1 - 1) // 2, c1 - 1)
            g = (s[0] + dr, s[1] + dc)
            b = (g[0] + dr, g[1] + dc)
            if not (0 <= b[0] < R and 0 <= b[1] < C):
                continue
            d = dist_to_edge(b[0], b[1], R, C)
            cands.append((d, gdir, s, g, b))
        if not cands:
            continue
        cands.sort(key=lambda x: x[0])
        if len(cands) >= 2 and rng.random() >= ALIGN_PROB:
            pool = cands[1:2]  # second-nearest only: differs from the naive
                                # target but stays close, so it rarely runs
                                # into a neighboring pit's reserved footprint
        else:
            pool = cands[:1]
        _, gdir, sill, gate, beyond = pool[rng.randrange(len(pool))]
        dr, dc = DIRS[gdir]
        # the escape corridor (a few cells past the gate) must also be clear of
        # every other pit's reserved footprint, or its calibration is unreliable.
        corridor_check = []
        br, bc = beyond
        for _ in range(12):
            corridor_check.append((br, bc))
            br += dr; bc += dc
            if not (0 <= br < R and 0 <= bc < C):
                break
        if any((r, c) in reserved for (r, c) in corridor_check):
            continue
        H0_beyond = dist_to_edge(beyond[0], beyond[1], R, C)
        P_sill = H0_beyond + MARGIN
        for (r, c) in pit_rect:
            d = abs(r - sill[0]) + abs(c - sill[1])
            H[r][c] = P_sill + d
            pit_cells.add((r, c))
            reserved.add((r, c))

        # the pit's parapet is SOLID (impassable, fixed stonework) on every side
        # except the one calibrated gate cell -- so the pit has exactly one
        # possible exit, and any path that does not thread that exact cell is
        # blocked outright, not just expensive. The gate's own height is now a
        # direct two-neighbor computation (sill, beyond) since its lateral
        # neighbors are all solid stone and drop out of the slope constraint.
        wall1 = set()
        for (r, c) in pit_rect:
            for ddr, ddc in DIRS:
                nr, nc = r + ddr, c + ddc
                if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in pit_rect:
                    wall1.add((nr, nc))
        wall = wall1 - {gate}
        for (r, c) in wall:
            obstacle[r][c] = True
            ring_cells.add((r, c))
            reserved.add((r, c))
        H[gate[0]][gate[1]] = min(P_sill + S, H0_beyond + S)
        reserved.add(gate)
        reserved.add(beyond)
        # keep a short straight escape corridor beyond the gate clear of future obstacles
        br, bc = beyond
        for _ in range(12):
            reserved.add((br, bc))
            br += dr; bc += dc
            if not (0 <= br < R and 0 <= bc < C):
                break
        gates.append((sill, gate, beyond, P_sill, H0_beyond))

    relax(H, obstacle, R, C, S)

    # ---- scatter a few single-cell obstacles in open terrain, well clear of every
    #      pit/ring/escape-corridor so they never interfere with a gate's calibration ----
    free = [(r, c) for r in range(2, R - 2) for c in range(2, C - 2)
            if (r, c) not in reserved]
    rng.shuffle(free)
    placed_obs = 0
    for (r, c) in free:
        if placed_obs >= nobstacles:
            break
        if all(not obstacle[r + dr][c + dc] for dr, dc in DIRS
               if 0 <= r + dr < R and 0 <= c + dc < C):
            obstacle[r][c] = True
            placed_obs += 1

    # ---- budget: verify (by simulation) which gates actually reconnect their
    #      pit once breached -- dense packing means a minority don't, since a
    #      neighboring pit's wall can intrude on the natural terrain a gate's
    #      calibration assumed -- and sum only the REAL minimal breach cost of
    #      the gates that work, with modest headroom. This keeps the budget
    #      tight around what the ridge-only strategy can actually achieve. ----
    maxsteps = R * C + 5
    total_breach = 0
    for (sill, gate, beyond, P_sill, H0_beyond) in gates:
        target = H0_beyond + 1
        cost = max(1, H[gate[0]][gate[1]] - target)
        saved = H[gate[0]][gate[1]]
        H[gate[0]][gate[1]] = target
        works = route(H, obstacle, gutter, R, C, sill, maxsteps)
        H[gate[0]][gate[1]] = saved
        if works:
            total_breach += cost
    B = int(round(total_breach * B_MULT)) + 2

    return R, C, S, B, H, obstacle, gutter, gates


def emit(R, C, S, B, H, obstacle, gutter):
    out = [f"{R} {C}", f"{S} {B}"]
    for r in range(R):
        out.append(' '.join(str(H[r][c]) for c in range(C)))
    for r in range(R):
        out.append(''.join('#' if obstacle[r][c] else '.' for c in range(C)))
    for r in range(R):
        out.append(''.join('G' if gutter[r][c] else '.' for c in range(C)))
    sys.stdout.write('\n'.join(out) + '\n')


def main():
    testId = int(sys.argv[1])
    R, C, S, B, H, obstacle, gutter, gates = build(testId)
    emit(R, C, S, B, H, obstacle, gutter)


if __name__ == "__main__":
    main()
