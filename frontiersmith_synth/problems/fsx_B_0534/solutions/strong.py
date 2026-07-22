# TIER: strong
# INSIGHT: because the arsonist's ignitions form a dense lattice hitting every
# compartment, the worst case equals the most-fuelled compartment.  So the job
# is not to ring hot spots or cut equal AREAS -- it is a balanced graph PARTITION:
# place the walls at fuel QUANTILES so every compartment carries equal fuel,
# driving the worst compartment down toward T / (#compartments).  We also search
# the split (how many vertical vs horizontal walls) the budget can afford.
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


def quantile_lines(prefix, total, k, free):
    # choose k cut lines so cumulative fuel crosses i/(k+1) * total.
    used = set(); chosen = []
    for i in range(1, k + 1):
        tgt_mass = i * total / (k + 1)
        # smallest boundary index b such that prefix[b] >= tgt_mass; wall sits
        # just before that column
        b = 0
        while b < len(prefix) - 1 and prefix[b] < tgt_mass:
            b += 1
        line = snap(b, free, used)
        if line is None:
            break
        used.add(line); chosen.append(line)
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

    colsum = [0] * N
    rowsum = [0] * N
    T = 0
    for r in range(N):
        for c in range(N):
            colsum[c] += fuel[r][c]
            rowsum[r] += fuel[r][c]
            T += fuel[r][c]
    # prefix[b] = fuel in columns 0..b-1
    cpref = [0] * (N + 1)
    for c in range(N):
        cpref[c + 1] = cpref[c] + colsum[c]
    rpref = [0] * (N + 1)
    for r in range(N):
        rpref[r + 1] = rpref[r] + rowsum[r]

    best = None
    for a in range(0, 5):
        for b in range(0, 5):
            cost = (a + b) * N - a * b
            if cost > F or (a == 0 and b == 0):
                continue
            vcols = quantile_lines(cpref, T, a, free_c)
            hrows = quantile_lines(rpref, T, b, free_r)
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
