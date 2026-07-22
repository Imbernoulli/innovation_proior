# TIER: trivial
# Reproduces the checker's baseline: a uniform conductance-1 backbone path over the
# active terminals. Ignores the targets entirely -> scores ~0.1.
import sys


def main():
    tok = sys.stdin.read().split()
    pos = 0
    n = int(tok[pos]); pos += 1
    m = int(tok[pos]); pos += 1
    wmax = float(tok[pos]); pos += 1
    P = int(tok[pos]); pos += 1
    verts = set()
    for _ in range(P):
        i = int(tok[pos]); j = int(tok[pos + 1]); pos += 4
        verts.add(i); verts.add(j)
    active = sorted(verts)
    edges = []
    for k in range(len(active) - 1):
        edges.append((active[k], active[k + 1], 1.0))
    out = [str(len(edges))]
    for (u, v, c) in edges:
        out.append("%d %d %.6f" % (u, v, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
