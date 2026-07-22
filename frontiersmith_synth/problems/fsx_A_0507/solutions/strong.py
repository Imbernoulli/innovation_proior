# TIER: strong
# INSIGHT: conduction from a source to a vent is a SERIES circuit -- the temperature is set
# by the worst bottleneck on the whole source->vent path, not by the cells nearest the source.
# So spend the budget building CONTIGUOUS CORRIDORS (shortest open path from a source to its
# nearest vent), not blankets. Rank sources by power*path_length (their resistance contribution)
# and fund the biggest reducers first; a corridor's harmonic-mean edges are ~KHi throughout,
# collapsing the dominant series resistance. Budget-limited -> this is a minimax allocation,
# left deliberately un-optimal (headroom above it).
import sys
from collections import deque

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    H = int(next(it)); W = int(next(it)); KHI = int(next(it)); K = int(next(it))
    S = int(next(it))
    sources = []
    for _ in range(S):
        r = int(next(it)); c = int(next(it)); p = int(next(it))
        sources.append((r, c, p))
    NV = int(next(it))
    vents = set()
    for _ in range(NV):
        r = int(next(it)); c = int(next(it)); vents.add((r, c))

    def is_wall(r, c):
        if r == 0 or r == H - 1 or c == 0 or c == W - 1:
            return (r, c) not in vents
        return False

    def path_to_vent(sr, sc):
        # BFS over open cells (interior + vents) to the nearest vent; return interior cells
        # on the path (source first, vent excluded).
        prev = {}
        q = deque([(sr, sc)])
        seen = {(sr, sc)}
        tgt = None
        while q:
            cur = q.popleft()
            if cur in vents:
                tgt = cur; break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if is_wall(nr, nc) or (nr, nc) in seen:
                    continue
                seen.add((nr, nc)); prev[(nr, nc)] = cur; q.append((nr, nc))
        if tgt is None:
            return []
        path = []
        cur = tgt
        while cur != (sr, sc):
            if cur not in vents:
                path.append(cur)
            cur = prev[cur]
        path.append((sr, sc))
        return path

    plans = []
    for (r, c, p) in sources:
        path = path_to_vent(r, c)
        plans.append((p * max(1, len(path)), path))
    plans.sort(key=lambda x: -x[0])

    up = []
    seen = set()
    for _, path in plans:
        for cell in path:            # source end first: a partial corridor still cuts resistance
            if len(up) >= K:
                break
            if cell in seen:
                continue
            seen.add(cell); up.append(cell)
        if len(up) >= K:
            break

    out = [str(len(up))]
    for (r, c) in up:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
