# TIER: strong
# Seeded multi-restart randomized greedy: try many shuffled activation orders under
# the corner-validity checker and keep the largest corner-free set. Fully deterministic
# (fixed seed schedule, no wall-clock).
import sys
import random


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    m = int(next(it))
    b = int(next(it))
    blocked = set()
    for _ in range(b):
        r = int(next(it)); c = int(next(it))
        blocked.add((r, c))

    cells = [(r, c) for r in range(m) for c in range(m) if (r, c) not in blocked]

    def ok(P, r, c):
        for d in range(1, m):
            if (r + d, c) in P and (r, c + d) in P:
                return False
        for d in range(1, m):
            if (r - d, c) in P and (r - d, c + d) in P:
                return False
        for d in range(1, m):
            if (r, c - d) in P and (r + d, c - d) in P:
                return False
        return True

    def build(order):
        P = set()
        for (r, c) in order:
            if ok(P, r, c):
                P.add((r, c))
        return P

    # candidate 1: deterministic row-major greedy
    best = build(sorted(cells))

    # candidate 2..: seeded shuffles
    restarts = 80
    for s in range(restarts):
        rng = random.Random(9001 + s)
        order = cells[:]
        rng.shuffle(order)
        P = build(order)
        if len(P) > len(best):
            best = P

    out = [str(len(best))]
    for (r, c) in sorted(best):
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
