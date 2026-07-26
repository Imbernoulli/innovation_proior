# TIER: strong
"""The insight: the droplet map is a watershed partition under a FIXED
deterministic descent rule, so almost everywhere an edit changes the surface
without changing anyone's destination -- it is provably wasted. Only at a
ridge cell (a boundary cell between a trapped basin and already-draining
terrain) can a single edit flip which side droplets fall to.

Every basin's members all converge, under the fixed descent rule, on ONE
lowest point (its sink) before getting stuck -- so the only edit that can
possibly rescue the whole basin is one that gives that sink a new exit, and
whether it truly works can only be confirmed by re-simulating from the sink.
For every basin we enumerate the sink's outside neighbors, compute (from the
max-slope bound alone) the cheapest height each could be edited down to, and
KEEP only the candidates that a re-simulation confirms actually reach a
gutter afterwards. Basins are then reconnected cheapest-ratio-first
(droplets rescued per unit budget), so the budget buys ridge flips, not
surface -- and never a plausible-looking edit that just relocates the puddle."""
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

    sink_members = {}
    for r in range(R):
        for c in range(C):
            if obstacle[r][c]:
                continue
            ok, end = route(H, obstacle, gutter, R, C, (r, c), maxsteps)
            if not ok:
                sink_members.setdefault(end, []).append((r, c))

    def neighbors(r, c):
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not obstacle[nr][nc]:
                yield nr, nc

    # For every basin, only the SINK (its unique lowest point, where every
    # member's descent ends) can matter: enumerate the sink's outside
    # neighbors, compute the cheapest slope-feasible target for each, and
    # keep only candidates a re-simulation confirms actually reach a gutter.
    plans = []
    for sink, members in sink_members.items():
        ar, ac = sink
        Ha = Hcur[ar][ac]
        candidates = []
        for (br, bc) in neighbors(ar, ac):
            lo = -10 ** 9
            hi = Ha - 1
            for (xr, xc) in neighbors(br, bc):
                xv = Hcur[xr][xc] if (xr, xc) != (ar, ac) else Ha
                lo = max(lo, xv - S)
                hi = min(hi, xv + S)
            if lo > hi:
                continue
            t = hi  # cheapest slope-feasible: the largest (least-edited) target
            cost = abs(Hcur[br][bc] - t)
            if cost == 0:
                continue
            candidates.append((cost, (br, bc), t))
        candidates.sort()
        best = None
        for cost, b, t in candidates:
            saved = Hcur[b[0]][b[1]]
            Hcur[b[0]][b[1]] = t
            ok, _ = route(Hcur, obstacle, gutter, R, C, sink, maxsteps)
            Hcur[b[0]][b[1]] = saved
            if ok:
                best = (cost, b, t)
                break
        if best is not None:
            plans.append((best[0], len(members), sink, best[1], best[2]))

    # reconnect the best droplets-per-budget basins first
    plans.sort(key=lambda p: -(p[1] / max(1, p[0])))

    used = 0
    for cost, size, sink, b, t in plans:
        br, bc = b
        # re-check feasibility (and that it still actually drains) against
        # the CURRENT heightfield before committing, since an earlier commit
        # may have touched a shared neighbor.
        lo = -10 ** 9
        hi = 10 ** 9
        for (xr, xc) in neighbors(br, bc):
            lo = max(lo, Hcur[xr][xc] - S)
            hi = min(hi, Hcur[xr][xc] + S)
        t2 = min(t, hi)
        if t2 < lo:
            continue
        real_cost = abs(Hcur[br][bc] - t2)
        if used + real_cost > B or real_cost == 0:
            continue
        saved = Hcur[br][bc]
        Hcur[br][bc] = t2
        ok, _ = route(Hcur, obstacle, gutter, R, C, sink, maxsteps)
        if not ok:
            Hcur[br][bc] = saved
            continue
        used += real_cost

    out = []
    for r in range(R):
        out.append(' '.join(str(Hcur[r][c]) for c in range(C)))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
