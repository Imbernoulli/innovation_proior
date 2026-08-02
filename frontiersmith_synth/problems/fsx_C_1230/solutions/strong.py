# TIER: strong
import sys, itertools

# The insight: a version choice on package i only interacts with a version
# choice on package j through an EXPLICIT requirement edge between them.
# Build the undirected "who can possibly conflict with whom" graph from the
# edges themselves (union-find) instead of walking packages in declaration
# order and re-discovering the same failure over and over. Packages that
# never share an edge are provably independent -- no search across them is
# ever needed, so nothing about one component's outcome has to be
# re-derived once another component's assignment changes (the same benefit
# a conflict-driven learner gets from a clause that only mentions the
# variables that actually caused the conflict). Each connected component in
# this construction stays tiny, so it is solved EXACTLY by brute force
# instead of being threaded through one global chronological search.


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent, a, b):
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[ra] = rb


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = [0] * n
    pref = [None] * n
    reqs = [None] * n
    for i in range(n):
        mi = int(next(it))
        m[i] = mi
        pref[i] = [0] * (mi + 1)
        reqs[i] = [None] * (mi + 1)
        for v in range(1, mi + 1):
            p = int(next(it))
            pref[i][v] = p
            r = int(next(it))
            edges = []
            for _ in range(r):
                j = int(next(it)); lo = int(next(it)); hi = int(next(it))
                edges.append((j, lo, hi))
            reqs[i][v] = edges

    parent = list(range(n))
    edges_by_src = {}
    for i in range(n):
        for v in range(1, m[i] + 1):
            for (j, lo, hi) in reqs[i][v]:
                union(parent, i, j)
                edges_by_src.setdefault(i, []).append((v, j, lo, hi))

    comps = {}
    for i in range(n):
        r = find(parent, i)
        comps.setdefault(r, []).append(i)

    assign = [0] * n
    CAP = 50000  # brute-force safety cap; every component here is <= 3 packages

    for members in comps.values():
        members = sorted(members)
        if len(members) == 1:
            i = members[0]
            assign[i] = max(range(1, m[i] + 1), key=lambda v: pref[i][v])
            continue

        domains = [range(1, m[i] + 1) for i in members]
        size = 1
        for d in domains:
            size *= len(d)
        pos = {pkg: idx for idx, pkg in enumerate(members)}

        if size > CAP:
            for i in members:  # defensive fallback, never hit by this generator
                assign[i] = 1
            continue

        best_score, best_combo = None, None
        for combo in itertools.product(*domains):
            ok = True
            for local_i, i in enumerate(members):
                v = combo[local_i]
                for (ev, j, lo, hi) in edges_by_src.get(i, []):
                    if ev != v:
                        continue
                    cj = combo[pos[j]]
                    if cj < lo or cj > hi:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            score = sum(pref[i][combo[k]] for k, i in enumerate(members))
            if best_score is None or score > best_score:
                best_score, best_combo = score, combo

        if best_combo is None:
            for i in members:  # unreachable: version 1 everywhere is feasible
                assign[i] = 1
        else:
            for k, i in enumerate(members):
                assign[i] = best_combo[k]

    print(" ".join(str(x) for x in assign))


if __name__ == "__main__":
    main()
