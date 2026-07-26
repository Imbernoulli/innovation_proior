"""Shared deterministic physics engine for the solidification-front problem.
Used by gen.py (to build an achievable target microstructure) and verify.py
(to score a submitted cooling schedule). NOT importable by solutions (they run
sandboxed with the synth tree hidden) -- solutions embed their own copy of the
step() logic inline.

Rules (must match statement.md exactly):
  - N cells in a line, each either LIQUID (has an integer heat value) or SOLID
    (locked forever, carries an integer orientation label).
  - A stage: (1) optionally cool ONE liquid cell by CSTEP (heat -= CSTEP);
             (2) every liquid cell updates synchronously via
                 new_heat[i] = floor((L + R + 2*heat[i]) / 4)
                 where L = heat[i-1] if cell i-1 exists and is liquid, else
                 heat[i] itself (no-flux / mirror at a solid or grid boundary);
                 R symmetric.
             (3) every liquid cell with new_heat <= F freezes SIMULTANEOUSLY;
                 each newly-frozen cell's orientation = orientation of the
                 NEAREST already-solid cell as of the START of the stage
                 (ties broken toward the LEFT neighbor); a cell with no solid
                 cell anywhere on the grid (impossible once >=1 seed exists)
                 gets orientation 0 (never matches a target in 1..M).
"""


def new_state(N, seeds, H0):
    """seeds: list of (pos, orientation). Returns (heat, solid, orient)."""
    heat = [H0] * N
    solid = [False] * N
    orient = [0] * N
    for pos, o in seeds:
        solid[pos] = True
        orient[pos] = o
    return heat, solid, orient


def step(heat, solid, orient, cool_idx, F, CSTEP):
    """Advance one stage in place. cool_idx is an int cell index or None.
    Returns True iff the requested cooling target was valid (liquid, in range);
    caller must check this BEFORE trusting the resulting state for scoring."""
    N = len(heat)
    if cool_idx is not None:
        if not (0 <= cool_idx < N) or solid[cool_idx]:
            return False
        heat[cool_idx] -= CSTEP
    old = heat[:]
    new = heat[:]
    for i in range(N):
        if solid[i]:
            continue
        L = old[i - 1] if (i - 1 >= 0 and not solid[i - 1]) else old[i]
        R = old[i + 1] if (i + 1 < N and not solid[i + 1]) else old[i]
        new[i] = (L + R + 2 * old[i]) // 4
    heat[:] = new

    newly = [i for i in range(N) if not solid[i] and heat[i] <= F]
    assign = {}
    for i in newly:
        dl = dr = None
        ol = orr = None
        j = i - 1
        while j >= 0:
            if solid[j]:
                dl = i - j
                ol = orient[j]
                break
            j -= 1
        j = i + 1
        while j < N:
            if solid[j]:
                dr = j - i
                orr = orient[j]
                break
            j += 1
        if dl is None and dr is None:
            assign[i] = 0
        elif dr is None or (dl is not None and dl <= dr):
            assign[i] = ol
        else:
            assign[i] = orr
    for i in newly:
        solid[i] = True
        orient[i] = assign[i]
    return True


def frontier_from_left(solid, seed_pos, N):
    """First liquid cell scanning rightward from seed_pos, or None if the
    contiguous solid run from seed_pos already reaches the grid edge."""
    i = seed_pos + 1
    while i < N and solid[i]:
        i += 1
    return i if i < N else None


def frontier_from_right(solid, seed_pos, N):
    i = seed_pos - 1
    while i >= 0 and solid[i]:
        i -= 1
    return i if i >= 0 else None
