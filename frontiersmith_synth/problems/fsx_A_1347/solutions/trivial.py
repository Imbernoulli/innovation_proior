# TIER: trivial
"""The naive, uncreative baseline: color each connected patch of the mesh
with a FRESH block of colors, never realizing that separate patches (which
share no vertex, hence no constraint) could safely reuse the very same
small palette. Cost-blind, no certificates. Reproduces the checker's own
internal reference construction."""
import sys


def components(adj, n):
    seen = [False] * (n + 1)
    comps = []
    for s in range(1, n + 1):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comp.sort()
        comps.append(comp)
    comps.sort(key=lambda c: c[0])
    return comps


def main():
    data = sys.stdin.read().split()
    ti = 0
    n = int(data[ti]); ti += 1
    m = int(data[ti]); ti += 1
    K = int(data[ti]); ti += 1
    ti += K  # trivial ignores costs
    adj = {v: set() for v in range(1, n + 1)}
    for _ in range(m):
        a, b, c = int(data[ti]), int(data[ti + 1]), int(data[ti + 2]); ti += 3
        for (u, v) in ((a, b), (b, c), (a, c)):
            adj[u].add(v)
            adj[v].add(u)

    comps = components(adj, n)
    color = [0] * (n + 1)
    block_start = 1
    for comp in comps:
        palette = [((block_start - 1 + i) % K) + 1 for i in range(K)]
        local = {}
        used_count = 0
        for v in comp:  # id order within the component
            used = {local[u] for u in adj[v] if u in local}
            for c in palette:
                if c not in used:
                    local[v] = c
                    break
            used_count += 1
        for v in comp:
            color[v] = local[v]
        distinct_here = len(set(local.values()))
        block_start = ((block_start - 1 + distinct_here) % K) + 1

    out = []
    out.append(" ".join(str(color[v]) for v in range(1, n + 1)))
    out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
