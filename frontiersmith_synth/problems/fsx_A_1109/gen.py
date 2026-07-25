import sys
import random

# gen.py <testId> -- prints ONE instance of the "metered crystal polisher" ledger.
#
# Instance: a sparse symmetric integer matrix A (n x n) with planted structure and a
# cost budget B.  Structure:
#   * "grit": small blocks (size 5..7) of banded (bandwidth 2) + a few extra random
#     internal edges, values 5..8 -- cheap pivots (low-degree rows), bulk of the energy.
#   * "bait": 1-2 hub nodes connected to ~60% of all rows with tiny values 2..4, of
#     which the first ~8 hub edges carry the LARGEST values in the matrix (9..11).
#     Any pivot touching a hub row costs ~0.6n (fill-in tax) and pollutes the partner
#     row to hub density, inflating every later pivot that touches it.
# Classical largest-pivot Jacobi runs straight at the bait and blows the budget on
# inferior energy-per-cost pivots; the strong schedule defers/ignores the bait and
# harvests cheap structure first.  Trap strength grows with testId (n grows, second
# hub appears at t>=4).
#
# Seeded by testId only; fully deterministic.

def main():
    t = int(sys.argv[1])
    rng = random.Random(910000 + 31337 * t)

    n = 30 + 2 * t                       # 32 .. 50
    bs = [5, 6, 7][(t - 1) % 3]          # block size
    bw = 2                               # bandwidth inside a block
    diag = [rng.randint(40, 90) for _ in range(n)]

    blocks = []
    s = 0
    while s < n:
        e = min(n, s + bs)
        blocks.append(list(range(s, e)))
        s = e

    entries = {}
    # grit: banded + extra random internal edges, values 5..8
    for blk in blocks:
        for a in range(len(blk)):
            for b in range(a + 1, len(blk)):
                if b - a <= bw:
                    entries[(blk[a], blk[b])] = rng.randint(5, 8)
        for _ in range(4):
            a = rng.randrange(len(blk))
            b = rng.randrange(len(blk))
            if a == b:
                continue
            i, j = min(blk[a], blk[b]), max(blk[a], blk[b])
            if (i, j) not in entries:
                entries[(i, j)] = rng.randint(5, 8)

    # hubs: dense moderate rows with a few large bait edges
    nh = 1 if t <= 3 else 2
    cand = list(range(n))
    rng.shuffle(cand)
    hubs = cand[:nh]
    for h in hubs:
        nbrs = [jx for jx in range(n) if jx != h and rng.random() < 0.6]
        rng.shuffle(nbrs)
        nbait = min(len(nbrs), 8)
        for k, jx in enumerate(nbrs):
            i, j = (h, jx) if h < jx else (jx, h)
            if k < nbait:
                entries[(i, j)] = rng.randint(9, 11)   # bait: largest values
            elif (i, j) not in entries:
                entries[(i, j)] = rng.randint(2, 4)    # dense moderate filler

    ents = [(i, j, v) for (i, j), v in sorted(entries.items())]
    B = 22 * n

    out = []
    out.append("%d %d %d" % (n, B, len(ents)))
    out.append(" ".join(str(d) for d in diag))
    for (i, j, v) in ents:
        out.append("%d %d %d" % (i + 1, j + 1, v))     # 1-indexed, i < j
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
