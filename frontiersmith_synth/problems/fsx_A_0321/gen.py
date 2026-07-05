#!/usr/bin/env python3
# gen.py <testId>  -- prints ONE vineyard-irrigation instance to stdout.
# testId 1..10 is a difficulty ladder (small dimension -> large dimension).
# Everything is seeded by testId only, so instances are deterministic.
#
# Instance semantics (see statement.md):
#   The vineyard is addressed by n ternary "row.terrace" coordinates, i.e. every
#   plot is a vector in F_3^n (3^n plots total).  A few plots are rocky (blocked)
#   and cannot host an emitter.  Every plot has an integer water yield.
import sys, random

# dimension ladder: small = fast reward, large = evaluation
NMAP = {1: 4, 2: 5, 3: 5, 4: 6, 5: 6, 6: 7, 7: 7, 8: 8, 9: 8, 10: 8}


def str_of(idx, n):
    d = []
    for _ in range(n):
        d.append(idx % 3)
        idx //= 3
    return "".join(str(x) for x in reversed(d))


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    n = NMAP.get(t, 6)
    N = 3 ** n
    seed = 100003 * t + 777
    rng = random.Random(seed)

    # blocked (rocky) plots: ~3% of the vineyard, never the all-zero plot
    m = max(1, int(0.03 * N))
    blocked = set()
    while len(blocked) < m:
        b = rng.randrange(N)
        if b != 0:
            blocked.add(b)
    blocked = sorted(blocked)

    # water yields for every plot, index order 0..N-1
    weights = [1 + rng.randrange(1000) for _ in range(N)]

    out = []
    out.append(str(n))
    out.append(str(len(blocked)))
    for b in blocked:
        out.append(str_of(b, n))
    # weights: 20 per line for readability
    line = []
    for i, w in enumerate(weights):
        line.append(str(w))
        if len(line) == 20:
            out.append(" ".join(line))
            line = []
    if line:
        out.append(" ".join(line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
