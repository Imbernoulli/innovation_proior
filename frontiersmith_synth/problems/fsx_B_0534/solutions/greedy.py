# TIER: greedy
# The obvious approach: partition the map into a grid of compartments with
# straight firebreak walls, placed at UNIFORM (equal-area) positions -- classic
# recursive-bisection thinking.  It never looks at WHERE the fuel actually sits,
# so on a skewed map one compartment inherits most of the fuel and the
# worst-case burn stays high.
import sys


def read_instance():
    it = sys.stdin.read().split()
    p = 0
    N = int(it[p]); F = int(it[p + 1]); K = int(it[p + 2]); p += 3
    ign = []
    ign_r = set(); ign_c = set()
    for _ in range(K):
        r = int(it[p]); c = int(it[p + 1]); p += 2
        ign.append((r, c)); ign_r.add(r); ign_c.add(c)
    fuel = [[0] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            fuel[r][c] = int(it[p]); p += 1
    return N, F, ign, ign_r, ign_c, fuel


def worst_burned(N, fuel, blocked, ign):
    parent = list(range(N * N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in range(N):
        base = r * N
        for c in range(N):
            i = base + c
            if blocked[i]:
                continue
            if c + 1 < N and not blocked[i + 1]:
                union(i, i + 1)
            if r + 1 < N and not blocked[i + N]:
                union(i, i + N)
    comp = {}
    for r in range(N):
        base = r * N
        for c in range(N):
            i = base + c
            if blocked[i]:
                continue
            root = find(i)
            comp[root] = comp.get(root, 0) + fuel[r][c]
    worst = 0
    for (r, c) in ign:
        v = comp.get(find(r * N + c), 0)
        if v > worst:
            worst = v
    return worst


def snap(target, free, used):
    best = None; bd = None
    for c in free:
        if c in used:
            continue
        d = abs(c - target)
        if bd is None or d < bd:
            bd = d; best = c
    return best


def pick_cols(N, a, free, uniform_targets):
    used = set(); chosen = []
    for tgt in uniform_targets:
        c = snap(tgt, free, used)
        if c is None:
            break
        used.add(c); chosen.append(c)
    return chosen


def build_blocked(N, vcols, hrows):
    blocked = [False] * (N * N)
    for c in vcols:
        for r in range(N):
            blocked[r * N + c] = True
    for r in hrows:
        base = r * N
        for c in range(N):
            blocked[base + c] = True
    return blocked


def main():
    N, F, ign, ign_r, ign_c, fuel = read_instance()
    free_c = [c for c in range(N) if c not in ign_c]
    free_r = [r for r in range(N) if r not in ign_r]

    best = None
    for a in range(0, 5):
        for b in range(0, 5):
            cost = (a + b) * N - a * b
            if cost > F or (a == 0 and b == 0):
                continue
            vt = [round((i + 1) * N / (a + 1)) for i in range(a)]
            ht = [round((i + 1) * N / (b + 1)) for i in range(b)]
            vcols = pick_cols(N, a, free_c, vt)
            hrows = pick_cols(N, b, free_r, ht)
            blocked = build_blocked(N, vcols, hrows)
            w = worst_burned(N, fuel, blocked, ign)
            if best is None or w < best[0]:
                best = (w, vcols, hrows)

    if best is None:
        sys.stdout.write("0\n"); return
    _, vcols, hrows = best
    cells = set()
    for c in vcols:
        for r in range(N):
            cells.add((r, c))
    for r in hrows:
        for c in range(N):
            cells.add((r, c))
    out = [str(len(cells))]
    for (r, c) in cells:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
