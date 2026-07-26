# TIER: greedy
"""The obvious first idea: find every cell where a droplet currently pools,
group them by where they pool (the basin), and for the biggest basins first,
carve a straight channel (an L-shaped path, row-then-column) from the puddle
to the nearest gutter -- setting every cell on that path to a strictly
decreasing height. This is a perfectly reasonable recipe and it does recover
some basins, but it pays for the ENTIRE path (often a dozen-plus cells) even
though most of that path was already fine; it also gives up outright if an
obstacle sits on its single chosen straight line, never trying another route.
It has no notion that only the ridge cell where the pool meets already-draining
terrain actually needs to move."""
import sys

DIRS = [(-1, 0), (0, -1), (1, 0), (0, 1)]


def read_instance():
    data = sys.stdin.read().split('\n')
    idx = 0
    R, C = map(int, data[idx].split()); idx += 1
    S, B = map(int, data[idx].split()); idx += 1
    H = []
    for _ in range(R):
        H.append(list(map(int, data[idx].split()))); idx += 1
    obstacle = []
    for _ in range(R):
        obstacle.append([ch == '#' for ch in data[idx]]); idx += 1
    gutter = []
    for _ in range(R):
        gutter.append([ch == 'G' for ch in data[idx]]); idx += 1
    return R, C, S, B, H, obstacle, gutter


def route(H, obstacle, gutter, R, C, start, maxsteps):
    pos = start
    for _ in range(maxsteps):
        r, c = pos
        if gutter[r][c]:
            return True, pos
        best = None
        bh = None
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not obstacle[nr][nc] and H[nr][nc] < H[r][c]:
                if bh is None or H[nr][nc] < bh:
                    bh = H[nr][nc]
                    best = (nr, nc)
        if best is None:
            return False, pos
        pos = best
    return False, pos


def main():
    R, C, S, B, H, obstacle, gutter = read_instance()
    maxsteps = R * C + 5
    Hcur = [row[:] for row in H]

    sink_count = {}
    for r in range(R):
        for c in range(C):
            if obstacle[r][c]:
                continue
            ok, end = route(H, obstacle, gutter, R, C, (r, c), maxsteps)
            if not ok:
                sink_count[end] = sink_count.get(end, 0) + 1

    gutter_cells = [(r, c) for r in range(R) for c in range(C) if gutter[r][c]]
    basins = sorted(sink_count.items(), key=lambda kv: -kv[1])

    used = 0
    for sink, _cnt in basins:
        best_g = min(gutter_cells, key=lambda g: abs(g[0] - sink[0]) + abs(g[1] - sink[1]))
        path = []
        r, c = sink
        while r != best_g[0]:
            r += 1 if best_g[0] > r else -1
            path.append((r, c))
        while c != best_g[1]:
            c += 1 if best_g[1] > c else -1
            path.append((r, c))
        if not path or path[-1] != best_g:
            continue
        if any(obstacle[pr][pc] for pr, pc in path):
            continue  # the straight line is blocked -- give up on this basin

        cand = {}
        v = Hcur[sink[0]][sink[1]]
        for (pr, pc) in path:
            v -= 1
            cand[(pr, pc)] = v
        cost = sum(abs(cand[p] - Hcur[p[0]][p[1]]) for p in path)
        if used + cost > B:
            continue

        def hval(p):
            return cand.get(p, Hcur[p[0]][p[1]])

        ok = True
        for (pr, pc) in path:
            for dr, dc in DIRS:
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < R and 0 <= nc < C and not obstacle[nr][nc]:
                    if abs(hval((pr, pc)) - hval((nr, nc))) > S:
                        ok = False
                        break
            if not ok:
                break
        if not ok:
            continue

        for p, val in cand.items():
            Hcur[p[0]][p[1]] = val
        used += cost

    out = []
    for r in range(R):
        out.append(' '.join(str(Hcur[r][c]) for c in range(C)))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
