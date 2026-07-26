# TIER: invalid
"""
Deliberately infeasible: claims more edges than the quota allows AND reweights an edge above
its original weight (dim-only is required). Must score 0.
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

    k_count = s + 5  # over quota
    chosen = edges[:k_count] if len(edges) >= k_count else (edges * ((k_count // max(1, len(edges))) + 1))[:k_count]

    out = [str(k_count)]
    for (u, v, w) in chosen:
        out.append(f"{u} {v} {w * 50.0 + 1000.0:.6f}")  # wildly boosted, way above original
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
