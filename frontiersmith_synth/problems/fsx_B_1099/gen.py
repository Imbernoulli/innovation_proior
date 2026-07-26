import sys

# A serpentine "climbing corridor": the vine starts at the bottom (row 0)
# and a stack of NB horizontal baffle walls forces it upward through a
# sequence of rooms, each baffle solid except a narrow gap that alternates
# ends (far-left / far-right) from one baffle to the next.
#
# Room 0 (the start room -- the only one whose entry is not pinned to a
# grid edge) plants a LOCAL "false lead": directly beside the entry, a
# one-cell-wide dead-end shaft climbs straight up, flanked by walls and
# capped by the room's own ceiling (solid everywhere except the real,
# offset gap). Pure gravitropism's very first move is "up", which the
# false lead satisfies immediately; once inside, the flanking walls plus
# self-avoidance make it a strict one-way trip -- the tip dies there,
# having gained nothing. Thigmotropism reads the asymmetric wall mass
# right at the entrance and points the OTHER way, toward the real
# corridor, before the tip ever commits to the shaft.
#
# Room 0 also plants a second, separate one-cell dead-end POCKET just off
# the real corridor, with its own (dimmer) light at the bottom: a lone tip
# has to choose between detouring into it (forfeiting the rest of the
# climb) or skipping it and continuing on. Spending one branch at the
# pocket's floor cell lets a forked tip collect it while the main tip
# keeps climbing -- the only way to get both.
#
# Light: one source at the very start (every policy's path includes it, so
# the checker's internal baseline is bounded and non-degenerate), the
# pocket's light, one decoy in room 0's real corridor (visible only once
# you have actually left the false lead), and one bright source at the
# very top, occluded by every baffle until the last gap is threaded.
#
# Deterministic per testId: the structural ladder (size, baffle count,
# budgets) is a fixed function of testId; no randomness is needed.

LADDER = {
    1:  dict(C=9,  NB=2, RH=3, K=2),
    2:  dict(C=11, NB=2, RH=3, K=2),
    3:  dict(C=11, NB=3, RH=3, K=3),
    4:  dict(C=13, NB=3, RH=4, K=3),
    5:  dict(C=13, NB=4, RH=4, K=3),
    6:  dict(C=15, NB=4, RH=4, K=4),
    7:  dict(C=15, NB=5, RH=4, K=4),
    8:  dict(C=17, NB=5, RH=5, K=4),
    9:  dict(C=17, NB=6, RH=5, K=5),
    10: dict(C=19, NB=6, RH=5, K=5),
}

START_BRIGHT = 20
DECOY_BRIGHT = 14
TOP_BRIGHT = 60


def build(testId):
    p = LADDER.get(testId, LADDER[1])
    C, NB, RH, K = p['C'], p['NB'], p['RH'], p['K']
    GW = 1

    room_ranges = []    # (lo, hi) inclusive per room, NB+1 rooms
    entry_cols = []     # entry column of room i (index 0-based)
    gap_cols = []

    start_col = C // 2
    r = 0
    lo = r
    r += RH
    room_ranges.append((lo, r - 1))
    entry_cols.append(start_col)

    for i in range(1, NB + 1):
        # baffle 1's gap is on the LEFT: room 0's false lead defaults to
        # the east side (see below), so its exit must be west, opposite
        # the trap, or the trap's flanking walls would block the very
        # sweep that has to find this gap.
        side = 'L' if (i % 2 == 1) else 'R'
        gcol = (C - 1) if side == 'R' else 0
        gap_cols.append(gcol)
        r += 1  # baffle row
        lo = r
        r += RH
        room_ranges.append((lo, r - 1))
        entry_cols.append(gcol)

    R = r
    rows = [['.'] * C for _ in range(R)]

    # carve the baffles
    baffle_rows = []
    br = room_ranges[0][1] + 1
    for i in range(NB):
        baffle_rows.append(br)
        gcol = gap_cols[i]
        for c in range(C):
            if not (gcol <= c < gcol + GW):
                rows[br][c] = '#'
        br = room_ranges[i + 1][1] + 1

    # carve a false-lead dead-end shaft: directly above the room's entry,
    # blocking straight "up"; flanked by a wall on its far side; capped
    # naturally by the room's own ceiling (solid everywhere except the
    # real, offset gap). Placed on the side OPPOSITE the room's own exit
    # gap, so its full-height flanking walls never block the sweep that
    # later has to reach that gap.
    def carve_trap(i):
        lo, hi = room_ranges[i]
        ecol = entry_cols[i]
        exit_gap_col = gap_cols[i]
        if exit_gap_col >= ecol:
            tcol, farcol = ecol - 1, ecol - 2
            if tcol < 0:
                tcol, farcol = ecol + 1, ecol + 2
        else:
            tcol, farcol = ecol + 1, ecol + 2
            if farcol > C - 1:
                tcol, farcol = ecol - 1, ecol - 2
        for rr in range(lo + 1, hi + 1):
            rows[rr][ecol] = '#'
            if 0 <= farcol < C:
                rows[rr][farcol] = '#'
        # tcol (the shaft interior) stays open by construction
        return tcol

    # Only room 0 (the one room whose entry is NOT pinned to a grid edge)
    # can host this false lead without its flanking walls ever blocking
    # the gap-search sweep of a LATER room (every other room's entry sits
    # exactly at a boundary column, so the only available shaft direction
    # is also the only available sweep direction -- there is no room left
    # to place the trap without it swallowing the real path too). Room 0
    # gets a visible decoy just past its false lead (on the real-corridor
    # side): enough for a photo-aware-but-touch-blind policy to do just as
    # well as a touch-aware one HERE. But room 0's own false lead is a
    # dead end regardless of how it is avoided, and once past it there is
    # nothing else to see until the very top -- so from here on, ANY
    # policy (blind, photo-only, or touch-integrated) that escaped the
    # false lead has to rely on the same cue-free tie-break to thread
    # every remaining alternating gap; the false lead itself is where
    # touch genuinely earns its keep.
    carve_trap(0)

    # A second, SEPARATE one-cell pocket sits just off the real corridor,
    # between the start and the decoy: a lone tip has to choose between
    # detouring into it (a one-way dead end, forfeiting the rest of the
    # climb) or skipping it and continuing on. Spending one branch at the
    # pocket's own floor cell lets a forked tip collect it while the main
    # tip keeps climbing -- the only way to get both, which is what makes
    # the branch budget genuinely pay for itself (rather than the naive
    # "branch at the brightest raw coordinate" recipe, which has no way to
    # know this cell -- not the pocket's own tip -- is the one that works).
    exit_gap_col = gap_cols[0]
    direction = 1 if exit_gap_col >= start_col else -1
    pocket_col = start_col + 2 * direction
    far_flank = pocket_col + direction
    near_flank = pocket_col - direction
    ok = 0 <= pocket_col < C and 0 <= far_flank < C and 0 <= near_flank < C and RH >= 3
    if ok:
        rows[1][far_flank] = '#'
        rows[1][near_flank] = '#'
        rows[1][pocket_col] = '.'   # the pocket cell itself
        rows[2][pocket_col] = '#'   # cap: a strict one-cell-deep dead end
        # flanked on both lateral sides and capped above, so the ONLY way
        # in/out of (1,pocket_col) is straight down to (0,pocket_col) --
        # a genuine one-way trip for a single, self-avoiding tip.
        bonus_cell = (1, pocket_col)
    else:
        bonus_cell = None

    rows[0][start_col] = 'S'

    sources = [(0, start_col, START_BRIGHT)]

    lo, hi = room_ranges[0]
    rr = (lo + hi) // 2
    decoy_col = max(0, min(C - 1, gap_cols[0]))
    sources.append((rr, decoy_col, DECOY_BRIGHT))
    if bonus_cell is not None:
        sources.append((bonus_cell[0], bonus_cell[1], DECOY_BRIGHT + 2))

    # main bright source at the very top room, offset toward the side that
    # forces one last lateral traverse
    top_lo, top_hi = room_ranges[-1]
    top_row = top_hi
    last_side = 'R' if (NB % 2 == 1) else 'L'
    top_col = 1 if last_side == 'R' else C - 2
    top_col = max(0, min(C - 1, top_col))
    if rows[top_row][top_col] not in ('#', 'S'):
        sources.append((top_row, top_col, TOP_BRIGHT))

    best_src = {}
    for (rr, cc, b) in sources:
        key = (rr, cc)
        if key not in best_src or best_src[key] < b:
            best_src[key] = b
    sources = sorted((rr, cc, b) for (rr, cc), b in best_src.items())

    # step budget: generous multiple of the efficient wall-following path
    # (vertical climb + a near-full lateral traverse per baffle).
    ideal = R + NB * (C + RH)
    STEPS = int(2.2 * ideal) + 15

    return R, C, STEPS, K, rows, sources


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    R, C, STEPS, K, rows, sources = build(testId)
    out = []
    out.append("%d %d" % (R, C))
    out.append("%d %d" % (STEPS, K))
    for row in rows:
        out.append("".join(row))
    out.append(str(len(sources)))
    for (r, c, b) in sources:
        out.append("%d %d %d" % (r, c, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
