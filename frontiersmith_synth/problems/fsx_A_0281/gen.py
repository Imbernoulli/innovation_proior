#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance of the "Aurora relay grid" problem.

Instance = a toroidal ice-grid Z_n x Z_n (odd n) with a deterministic set of
"crevasse" cells that cannot host a relay.  testId 1..10 is a difficulty ladder
(small/fast n -> large/eval n).  Everything is seeded by testId only.

Output format (stdin the solver reads):
    n
    k
    x_1 y_1
    ...
    x_k y_k          (k blocked/crevasse cells, 0 <= x,y < n)
"""
import sys, random

# odd side lengths only: odd n => no degenerate corner (d = n/2) so the
# difference-set product construction is genuinely corner-free.
N_LADDER   = [9, 13, 15, 17, 21, 23, 27, 31, 33, 35]
DENS_LADDER = [0.030, 0.035, 0.040, 0.040, 0.045,
               0.045, 0.050, 0.055, 0.055, 0.060]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    t = max(1, min(len(N_LADDER), t))
    n = N_LADDER[t - 1]
    dens = DENS_LADDER[t - 1]

    rnd = random.Random(1000003 * t + 7)
    k_target = int(round(dens * n * n))
    blocked = set()
    guard = 0
    while len(blocked) < k_target and guard < 100000:
        guard += 1
        blocked.add((rnd.randrange(n), rnd.randrange(n)))

    # never block an entire row (keeps the single-row baseline sane)
    for y in range(n):
        rowfull = all((x, y) in blocked for x in range(n))
        if rowfull:
            blocked.discard((rnd.randrange(n), y))

    out = [str(n), str(len(blocked))]
    for (x, y) in sorted(blocked):
        out.append("%d %d" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
