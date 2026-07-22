# TIER: greedy
# The obvious recipe: connect all active terminals with a light backbone tree, then
# spend the remaining edge budget giving the most important target pairs their OWN
# direct wire set to conductance 1/target. NO awareness that every wire lowers the
# resistance of ALL pairs (Rayleigh monotonicity), so it systematically OVERSHOOTS
# (realized resistances come out below target) on the coupled clusters.
import sys


def main():
    tok = sys.stdin.read().split()
    pos = 0
    n = int(tok[pos]); pos += 1
    m = int(tok[pos]); pos += 1
    wmax = float(tok[pos]); pos += 1
    P = int(tok[pos]); pos += 1
    pt = []
    for _ in range(P):
        i = int(tok[pos]); j = int(tok[pos + 1])
        t = float(tok[pos + 2]); w = float(tok[pos + 3]); pos += 4
        a, b = (i, j) if i < j else (j, i)
        pt.append((a, b, t, w))

    active = sorted(set([a for (a, b, t, w) in pt] + [b for (a, b, t, w) in pt]))

    eset = set()
    edges = []

    def add(u, v, c):
        a, b = (u, v) if u < v else (v, u)
        if a == b or (a, b) in eset:
            return False
        eset.add((a, b))
        edges.append([a, b, c])
        return True

    # backbone spanning path (guarantees connectivity)
    g0 = 1.0
    for k in range(len(active) - 1):
        add(active[k], active[k + 1], g0)

    # remaining budget -> direct wires for most important pairs, c = 1/target
    remaining = m - len(edges)
    order = sorted(range(P), key=lambda k: -pt[k][3])
    for k in order:
        if remaining <= 0:
            break
        a, b, t, w = pt[k]
        c = min(wmax, 1.0 / t)
        if add(a, b, c):
            remaining -= 1

    out = [str(len(edges))]
    for (u, v, c) in edges:
        out.append("%d %d %.6f" % (u, v, min(wmax, max(1e-9, c))))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
