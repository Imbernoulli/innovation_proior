import sys, math

# ---------------------------------------------------------------------------
# Vine growth simulator (the SOLE authoritative copy). The participant's
# artifact is just three tropism weights (gravi/photo/thigmo) plus a set of
# up to K "branch-trigger" cells; THIS module deterministically grows the
# vine tip-by-tip, integrating the three cues at every step, and scores the
# total illumination of every distinct cell the vine occupies.
# ---------------------------------------------------------------------------

MAX_W = 1000.0          # sanity bound on a submitted weight
MAX_KP = 1000            # sanity bound on declared branch count (also capped by K)

# Movement is 4-connected (N,E,S,W); this fixed order is the tie-break
# priority for "best direction" (also what makes a thigmo-blind fallback
# default to a rightward drift on ties). Touch-sensing (thigmo) still looks
# at the full 8-neighbourhood so it can read a diagonally-adjacent wall.
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
UNIT = [(1.0 * dr, 1.0 * dc) for (dr, dc) in DIRS]
WALL_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

GRAVI = (1.0, 0.0)  # constant "grow away from gravity" = toward increasing row


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


# ---------------- instance parsing ----------------

def parse_instance(path):
    text = open(path, "r").read().split("\n")
    R, C = map(int, text[0].split())
    STEPS, K = map(int, text[1].split())
    grid = [list(text[2 + i]) for i in range(R)]
    start = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 'S':
                start = (r, c)
                grid[r][c] = '.'
    M = int(text[2 + R])
    sources = []
    for i in range(M):
        r, c, b = text[2 + R + 1 + i].split()
        sources.append((int(r), int(c), float(b)))
    return R, C, STEPS, K, grid, start, sources


# ---------------- static geometric fields ----------------

def bresenham_cells(r0, c0, r1, c1):
    cells = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def line_of_sight(grid, r0, c0, r1, c1):
    cells = bresenham_cells(r0, c0, r1, c1)
    for (r, c) in cells[1:-1]:
        if grid[r][c] == '#':
            return False
    return True


def rotate_cw(dr, dc):
    # rotate a (row,col) offset 90 degrees clockwise in (row=up, col=right)
    return (-dc, dr)


def build_fields(R, C, grid, sources):
    """illum/vis/thigmo are purely geometric (static). `vis[(r,c)]` is the
    list of source-indices with unobstructed line-of-sight from (r,c) --
    photo_vec itself is computed DYNAMICALLY during simulate() from `vis`
    plus which sources have already been reached ("consumed": once the vine
    occupies a source's own cell, that source stops pulling -- it has been
    captured, so phototropism does not yank the tip back through it)."""
    illum = {}
    vis = {}
    thigmo = {}
    for r in range(R):
        for c in range(C):
            if grid[r][c] == '#':
                continue
            ill = 0.0
            seen = []
            for idx, (sr, sc, sb) in enumerate(sources):
                if line_of_sight(grid, r, c, sr, sc):
                    dist = math.hypot(sr - r, sc - c)
                    ill += sb / (1.0 + dist)
                    seen.append(idx)
            illum[(r, c)] = ill
            vis[(r, c)] = seen

            normal = [0, 0]
            for (dr, dc) in WALL_DIRS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < R and 0 <= nc < C) or grid[nr][nc] == '#':
                    normal[0] += dr
                    normal[1] += dc
            if normal[0] == 0 and normal[1] == 0:
                thigmo[(r, c)] = (0.0, 0.0)
            else:
                thigmo[(r, c)] = rotate_cw(normal[0], normal[1])
    return illum, vis, thigmo


def photo_vec_at(r, c, sources, vis, consumed):
    # direction toward every visible, not-yet-consumed source, weighted by
    # brightness/(1+distance) -- same falloff as illum, so a source's pull
    # softens with range instead of competing at full strength from afar.
    pr = pc = 0.0
    for idx in vis[(r, c)]:
        if consumed[idx]:
            continue
        sr, sc, sb = sources[idx]
        dist = math.hypot(sr - r, sc - c)
        if dist > 1e-9:
            w = sb / (1.0 + dist)
            pr += w * (sr - r) / dist
            pc += w * (sc - c) / dist
    return pr, pc


# ---------------- the simulator ----------------

def best_and_second(V, r, c, R, C, grid, occupied):
    cands = []
    for k, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#' and (nr, nc) not in occupied:
            ur, uc = UNIT[k]
            dot = V[0] * ur + V[1] * uc
            cands.append((dot, (nr, nc)))
    if not cands:
        return None, None
    best = max(cands, key=lambda x: x[0])
    cands.remove(best)
    second = max(cands, key=lambda x: x[0]) if cands else None
    return best, second


def simulate(R, C, grid, start, STEPS, K, wg, wp, wt, branch_cells, illum, vis, thigmo, sources):
    occupied = {start}
    tips = [{'pos': start, 'alive': True, 'first_secondary': False}]
    branch_triggered = set()
    branches_used = 0
    steps_remaining = STEPS
    consumed = [False] * len(sources)
    src_at = {}
    for idx, (sr, sc, sb) in enumerate(sources):
        src_at[(sr, sc)] = idx
    if start in src_at:
        consumed[src_at[start]] = True

    while steps_remaining > 0:
        active = [t for t in tips if t['alive']]
        if not active:
            break
        made_progress = False
        for t in active:
            if steps_remaining <= 0:
                break
            r, c = t['pos']
            pr, pc = photo_vec_at(r, c, sources, vis, consumed)
            tr, tc = thigmo[(r, c)]
            V = (wg * GRAVI[0] + wp * pr + wt * tr, wg * GRAVI[1] + wp * pc + wt * tc)
            best, second = best_and_second(V, r, c, R, C, grid, occupied)
            if best is None:
                t['alive'] = False
                continue
            if t['first_secondary'] and second is not None:
                target = second[1]
            else:
                target = best[1]
            t['first_secondary'] = False
            occupied.add(target)
            t['pos'] = target
            steps_remaining -= 1
            made_progress = True
            if target in src_at:
                consumed[src_at[target]] = True
            if target in branch_cells and target not in branch_triggered and branches_used < K:
                branch_triggered.add(target)
                branches_used += 1
                tips.append({'pos': target, 'alive': True, 'first_secondary': True})
        if not made_progress:
            break

    F = captured_light(occupied, sources, vis)
    return F


def captured_light(occupied, sources, vis):
    """Score = sum over sources of the BEST illumination any single visited
    cell achieves for that source (not a per-cell sum). This is what makes
    "captured light" mean genuinely reaching toward each source rather than
    padding the score by wandering many mediocre cells near one bright
    cluster -- lingering near an already-approached source earns nothing
    further."""
    best_per_source = [0.0] * len(sources)
    for (r, c) in occupied:
        for idx in vis[(r, c)]:
            sr, sc, sb = sources[idx]
            dist = math.hypot(sr - r, sc - c)
            val = sb / (1.0 + dist)
            if val > best_per_source[idx]:
                best_per_source[idx] = val
    return sum(best_per_source)


# ---------------- internal baseline: pure gravitropism, no branches ----------------

def compute_baseline_F(R, C, grid, start, STEPS, K, illum, vis, thigmo, sources):
    return simulate(R, C, grid, start, STEPS, K, 1.0, 0.0, 0.0, frozenset(), illum, vis, thigmo, sources)


# ---------------- main ----------------

def main():
    R, C, STEPS, K, grid, start, sources = parse_instance(sys.argv[1])
    illum, vis, thigmo = build_fields(R, C, grid, sources)

    out_text = open(sys.argv[2]).read()
    tokens = out_text.split()
    idx = 0

    def next_tok():
        nonlocal idx
        if idx >= len(tokens):
            fail("truncated output")
        v = tokens[idx]
        idx += 1
        return v

    try:
        wg = float(next_tok())
        wp = float(next_tok())
        wt = float(next_tok())
    except ValueError:
        fail("non-numeric weight token")

    for name, w in (("gravi", wg), ("photo", wp), ("thigmo", wt)):
        if not math.isfinite(w):
            fail("non-finite %s weight" % name)
        if w < 0.0:
            fail("%s weight negative" % name)
        if w > MAX_W:
            fail("%s weight out of range" % name)
    if wg + wp + wt <= 1e-12:
        fail("all tropism weights are zero")

    try:
        Kp = int(next_tok())
    except ValueError:
        fail("non-integer branch count")
    if Kp < 0 or Kp > MAX_KP or Kp > K:
        fail("branch count out of range")

    branch_cells = set()
    for _ in range(Kp):
        try:
            r = int(next_tok())
            c = int(next_tok())
        except ValueError:
            fail("non-integer branch cell coordinate")
        if not (0 <= r < R and 0 <= c < C):
            fail("branch cell out of bounds")
        if grid[r][c] == '#':
            fail("branch cell is a wall")
        if (r, c) in branch_cells:
            fail("duplicate branch cell")
        branch_cells.add((r, c))

    if idx != len(tokens):
        fail("trailing garbage after declared tokens")

    F = simulate(R, C, grid, start, STEPS, K, wg, wp, wt, branch_cells, illum, vis, thigmo, sources)
    if not math.isfinite(F):
        fail("non-finite objective")

    B = compute_baseline_F(R, C, grid, start, STEPS, K, illum, vis, thigmo, sources)
    if not math.isfinite(B) or B <= 1e-9:
        fail("internal baseline degenerate")

    sc = min(1000.0, 100.0 * F / B)
    sc = max(0.0, sc)
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
