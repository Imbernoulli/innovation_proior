# TIER: strong
# The insight: light alone is not a reliable cue once it is occluded, and a
# fixed direction tie-break is not a substitute for real touch-following, so
# no SINGLE dominant tropism works everywhere. Instead:
#   (1) integrate all three cues at once and search a small, fixed grid of
#       (gravi, photo, thigmo) weight triples -- balanced weight vectors let
#       thigmo silently take over exactly on the cells where photo goes to
#       zero (occluded), which is what actually threads a zig-zag gap;
#   (2) structurally locate the maze's real bottlenecks (the sparse open
#       cells inside each mostly-solid wall row = the gaps) instead of
#       aiming branches at raw source coordinates, and spend the branch
#       budget on those bottlenecks via greedy forward selection, re-scoring
#       with a full re-simulation each time.
# Both the weight grid and the forward-selection search are fixed-size and
# deterministic (no randomness, no wall-clock cutoffs).
import sys, math, itertools

DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
UNIT = [(1.0 * dr, 1.0 * dc) for (dr, dc) in DIRS]
WALL_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
GRAVI = (1.0, 0.0)


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
    for (r, c) in bresenham_cells(r0, c0, r1, c1)[1:-1]:
        if grid[r][c] == '#':
            return False
    return True


def rotate_cw(dr, dc):
    return (-dc, dr)


def build_fields(R, C, grid, sources):
    """illum/vis/thigmo are static geometry; photo_vec is computed
    dynamically in simulate() from `vis` + which sources are already
    consumed (occupied by the vine)."""
    illum, vis, thigmo = {}, {}, {}
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
            thigmo[(r, c)] = (0.0, 0.0) if normal == [0, 0] else rotate_cw(*normal)
    return illum, vis, thigmo


def photo_vec_at(r, c, sources, vis, consumed):
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


def best_and_second(V, r, c, R, C, grid, occupied):
    cands = []
    for k, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#' and (nr, nc) not in occupied:
            ur, uc = UNIT[k]
            cands.append((V[0] * ur + V[1] * uc, (nr, nc)))
    if not cands:
        return None
    return max(cands, key=lambda x: x[0])[1]


def simulate(R, C, grid, start, STEPS, K, wg, wp, wt, branch_cells, illum, vis, thigmo, sources):
    occupied = {start}
    tips = [{'pos': start, 'alive': True, 'first_secondary': False}]
    branch_triggered = set()
    branches_used = 0
    steps_remaining = STEPS
    consumed = [False] * len(sources)
    src_at = {(sr, sc): idx for idx, (sr, sc, sb) in enumerate(sources)}
    if start in src_at:
        consumed[src_at[start]] = True
    while steps_remaining > 0:
        active = [t for t in tips if t['alive']]
        if not active:
            break
        progressed = False
        for t in active:
            if steps_remaining <= 0:
                break
            r, c = t['pos']
            pr, pc = photo_vec_at(r, c, sources, vis, consumed)
            tr, tc = thigmo[(r, c)]
            V = (wg * GRAVI[0] + wp * pr + wt * tr, wg * GRAVI[1] + wp * pc + wt * tc)
            cands = []
            for k, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#' and (nr, nc) not in occupied:
                    ur, uc = UNIT[k]
                    cands.append((V[0] * ur + V[1] * uc, (nr, nc)))
            if not cands:
                t['alive'] = False
                continue
            best = max(cands, key=lambda x: x[0])
            if t['first_secondary']:
                rest = [x for x in cands if x != best]
                target = max(rest, key=lambda x: x[0])[1] if rest else best[1]
            else:
                target = best[1]
            t['first_secondary'] = False
            occupied.add(target)
            t['pos'] = target
            steps_remaining -= 1
            progressed = True
            if target in src_at:
                consumed[src_at[target]] = True
            if target in branch_cells and target not in branch_triggered and branches_used < K:
                branch_triggered.add(target)
                branches_used += 1
                tips.append({'pos': target, 'alive': True, 'first_secondary': True})
        if not progressed:
            break
    return captured_light(occupied, sources, vis)


def captured_light(occupied, sources, vis):
    best_per_source = [0.0] * len(sources)
    for (r, c) in occupied:
        for idx in vis[(r, c)]:
            sr, sc, sb = sources[idx]
            dist = math.hypot(sr - r, sc - c)
            val = sb / (1.0 + dist)
            if val > best_per_source[idx]:
                best_per_source[idx] = val
    return sum(best_per_source)


def find_branch_candidates(R, C, grid):
    """Structural bottlenecks worth a branch trigger: (a) baffle gap cells
    (mostly-solid rows), and (b) each room's floor row (the row right after
    a baffle, or row 0) -- that is where a room's entrance sits, and where
    a false-lead dead-end shaft forks off the real corridor, so it is
    exactly where a second tip should peel off to collect a dead-end's
    light while the first tip keeps climbing."""
    baffle_rows = [r for r in range(R) if sum(1 for c in range(C) if grid[r][c] == '#') > C // 2]
    cands = set()
    for r in baffle_rows:
        for c in range(C):
            if grid[r][c] != '#':
                cands.add((r, c))
    floor_rows = [0] + [r + 1 for r in baffle_rows if r + 1 < R]
    for r in floor_rows:
        for c in range(C):
            if grid[r][c] != '#':
                cands.add((r, c))
    return sorted(cands)


def main():
    data = sys.stdin.read().split("\n")
    R, C = map(int, data[0].split())
    STEPS, K = map(int, data[1].split())
    grid = [list(data[2 + i]) for i in range(R)]
    start = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 'S':
                start = (r, c)
                grid[r][c] = '.'
    M = int(data[2 + R])
    sources = []
    for i in range(M):
        r, c, b = data[2 + R + 1 + i].split()
        sources.append((int(r), int(c), float(b)))

    illum, vis, thigmo = build_fields(R, C, grid, sources)

    W_G = [0.0, 0.5, 1.0, 1.5]
    W_P = [0.0, 0.5, 1.0, 2.0]
    W_T = [0.0, 0.5, 1.0, 2.0, 3.0]

    # among (near-)tied weight triples, prefer one where all three cues are
    # genuinely active (integration), not just whichever a fixed iteration
    # order happens to reach first
    best_F, best_w = -1.0, (1.0, 0.0, 0.0)
    for wg, wp, wt in itertools.product(W_G, W_P, W_T):
        if wg + wp + wt <= 1e-12:
            continue
        F = simulate(R, C, grid, start, STEPS, K, wg, wp, wt, frozenset(), illum, vis, thigmo, sources)
        cur_mixed = min(wg, wp, wt) > 1e-9
        best_mixed = min(best_w) > 1e-9
        if F > best_F + 1e-9 or (F > best_F - 1e-9 and cur_mixed and not best_mixed):
            best_F, best_w = F, (wg, wp, wt)

    wg, wp, wt = best_w
    candidates = find_branch_candidates(R, C, grid)
    chosen = set()
    cur_F = simulate(R, C, grid, start, STEPS, K, wg, wp, wt, frozenset(chosen), illum, vis, thigmo, sources)
    for _ in range(K):
        best_gain, best_cell = 0.0, None
        for cell in candidates:
            if cell in chosen:
                continue
            trial = chosen | {cell}
            F = simulate(R, C, grid, start, STEPS, K, wg, wp, wt, frozenset(trial), illum, vis, thigmo, sources)
            if F - cur_F > best_gain:
                best_gain, best_cell = F - cur_F, cell
        if best_cell is None:
            break
        chosen.add(best_cell)
        cur_F += best_gain

    print("%.4f %.4f %.4f" % (wg, wp, wt))
    print(len(chosen))
    for (r, c) in sorted(chosen):
        print(r, c)


if __name__ == "__main__":
    main()
