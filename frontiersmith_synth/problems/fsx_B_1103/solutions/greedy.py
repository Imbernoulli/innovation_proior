# TIER: greedy
"""The obvious recipe: color changes are the scary cost, so lock to ONE color
at a time (the absolute minimum number of loads is C) and chain the
monochromatic trails with nearest-neighbor jumps.

Per color: decompose that color's subgraph into edge-disjoint trails
(Hierholzer, odd-degree starts first), then visit the trails in
nearest-neighbor order from the current needle position, jumping to the nearer
endpoint of the next trail.

This cannot see the planted orbit structure: every color's arcs are scattered
across ALL the far-apart motif copies, so each of the C color sweeps pays a
full lattice crossing of long jumps. Re-buying color changes along one short
region tour -- which this recipe refuses to do -- is far cheaper."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0

    def ni():
        nonlocal p
        v = int(data[p]); p += 1
        return v

    V = ni(); E = ni(); C = ni(); K = ni()
    xs = [0] * V; ys = [0] * V
    for i in range(V):
        xs[i] = ni(); ys[i] = ni()
    edges = [(ni(), ni(), ni()) for _ in range(E)]

    def dist(a, b):
        return abs(xs[a] - xs[b]) + abs(ys[a] - ys[b])

    by_color = [[] for _ in range(C)]
    for idx, (u, v, c) in enumerate(edges):
        by_color[c].append((u, v, idx))

    ops = []
    pos = 0
    for c in range(C):
        es = by_color[c]
        if not es:
            continue
        adj = {}
        for (u, v, idx) in es:
            adj.setdefault(u, []).append((v, idx))
            adj.setdefault(v, []).append((u, idx))
        for w in adj:
            adj[w].sort()
        used = set()
        trails = []
        starts = sorted(adj, key=lambda x: (len(adj[x]) % 2 == 0, x))
        for s in starts:
            while True:
                # any unused edge at s?
                if all(idx in used for (_, idx) in adj[s]):
                    break
                seq = [s]
                cur = s
                while True:
                    nxt = -1; nidx = -1
                    for (w, idx) in adj[cur]:
                        if idx not in used:
                            nxt = w; nidx = idx
                            break
                    if nxt < 0:
                        break
                    used.add(nidx)
                    seq.append(nxt)
                    cur = nxt
                trails.append(seq)
        # nearest-neighbor chaining of this color's trails
        remaining = list(trails)
        while remaining:
            best = None
            for ti, seq in enumerate(remaining):
                for end in (0, len(seq) - 1):
                    d = dist(pos, seq[end])
                    if best is None or (d, seq[end], ti) < best:
                        best = (d, seq[end], ti)
            d, target, ti = best
            seq = remaining.pop(ti)
            if seq[0] != target:
                seq = seq[::-1]
            if pos != seq[0]:
                ops.append("J %d" % seq[0])
                pos = seq[0]
            for k in range(len(seq) - 1):
                ops.append("S %d %d" % (seq[k], seq[k + 1]))
            pos = seq[-1]
    out = [str(len(ops))]
    out.extend(ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
