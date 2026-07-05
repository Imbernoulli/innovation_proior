#!/usr/bin/env python3
"""gen.py <testId> -- print ONE forest-fire watchtower channel-assignment instance.

An n x n grid of watchtower sites. Every tower must be tuned to one of n radio
CHANNELS 0..n-1. To avoid interference the fire authority enforces THREE rules:
no channel repeats within a grid ROW, within a grid COLUMN, or within a
frequency-reuse SECTOR. The n sectors partition the grid into n size-n groups,
so this is a Latin square with an extra region ("gerechte") constraint --
strictly harder than a plain Latin square.

Construction (odd n) guarantees a full interference-free tuning EXISTS:
  pick random permutations rho, gam, pi on Z_n and set the hidden full solution
      L(i,j) = pi[(rho[i] + gam[j]) mod n]
  with sectors
      sector(i,j) = (gam[j] - rho[i]) mod n .
  Each row/column is a Latin line by construction, and for ODD n each sector's
  channels are { pi[(2*rho[i] + r) mod n] : i } = all n channels, so (L, sector)
  is a valid gerechte design. A density-fraction of L's cells is then revealed
  as pre-tuned givens; the rest are blanked. Because a full completion provably
  exists, a naive greedy dead-ends while smarter search recovers far more.

Difficulty (n, given density) grows with testId. Everything is seeded by testId
only, so instances are bit-for-bit reproducible.
"""
import sys, random


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20260702 + 100003 * tid)

    # difficulty ladder: (n, reveal_density). "large" scale => odd n in [9..13].
    specs = [
        (9, 0.40), (9, 0.45), (11, 0.40), (11, 0.45), (11, 0.48),
        (13, 0.40), (13, 0.44), (13, 0.46), (13, 0.48), (9, 0.48),
    ]
    n, dens = specs[(tid - 1) % len(specs)]
    assert n % 2 == 1, "construction requires odd n"

    rho = list(range(n)); rng.shuffle(rho)
    gam = list(range(n)); rng.shuffle(gam)
    pi = list(range(n)); rng.shuffle(pi)

    full = [[pi[(rho[i] + gam[j]) % n] for j in range(n)] for i in range(n)]
    sect = [[(gam[j] - rho[i]) % n for j in range(n)] for i in range(n)]

    cells = [(i, j) for i in range(n) for j in range(n)]
    rng.shuffle(cells)
    k = max(1, int(round(dens * n * n)))
    given = [[None] * n for _ in range(n)]
    for (i, j) in cells[:k]:
        given[i][j] = full[i][j]

    out = [str(n)]
    for i in range(n):
        out.append(" ".join(str(sect[i][j]) for j in range(n)))
    for i in range(n):
        out.append(" ".join("." if given[i][j] is None else str(given[i][j])
                            for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
