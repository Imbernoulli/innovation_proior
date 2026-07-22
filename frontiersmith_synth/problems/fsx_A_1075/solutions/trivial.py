# TIER: trivial
"""Naive per-query left-to-right fold. Every query is solved completely independently:
no attempt is made to notice that other queries touch the same sub-range, and even the
per-query parenthesization is the dumbest possible (strict left fold), mirroring the
checker's own baseline construction."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    m = int(data[p]); p += 1
    dims = [int(data[p + i]) for i in range(m + 1)]; p += m + 1
    Q = int(data[p]); p += 1
    queries = []
    for _ in range(Q):
        L = int(data[p]); R = int(data[p + 1]); p += 2
        queries.append((L, R))

    nodes = []

    def leaf(i):
        nodes.append("L %d" % i)
        return len(nodes) - 1

    def split(c1, c2):
        nodes.append("S %d %d" % (c1, c2))
        return len(nodes) - 1

    roots = []
    for (L, R) in queries:
        cur = leaf(L)
        for t in range(L + 1, R):
            cur = split(cur, leaf(t))
        roots.append(cur)

    out = [str(len(nodes))]
    out.extend(nodes)
    out.append(" ".join(map(str, roots)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
