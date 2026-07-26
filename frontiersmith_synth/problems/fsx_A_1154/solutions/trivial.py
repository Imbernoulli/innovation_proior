# TIER: trivial
"""
Trivial baseline: keep the s heaviest edges of G, unchanged weight. Ignores the published
test-vector family entirely -- this is exactly the checker's internal baseline construction,
so it should score close to the ~0.1 reference point.
"""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0
    n = int(data[pos]); m = int(data[pos + 1]); s = int(data[pos + 2]); K = int(data[pos + 3])
    pos += 4
    edges = []
    for _ in range(m):
        u = int(data[pos]); v = int(data[pos + 1]); w = float(data[pos + 2]); pos += 3
        edges.append((u, v, w))
    # (test vectors are ignored by this tier)

    order = sorted(range(m), key=lambda i: -edges[i][2])
    sel = [edges[i] for i in order[:s]]

    out = [str(len(sel))]
    for (u, v, w) in sel:
        out.append(f"{u} {v} {w:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
