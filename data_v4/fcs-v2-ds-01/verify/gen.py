#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
# Prints: n q / array / q lines "l r" (1-based inclusive).
import sys
import random


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rnd = random.Random(seed)

    # Mix of regimes to stress add/remove deltas and the Hilbert ordering.
    regime = rnd.randint(0, 3)
    if regime == 0:
        n = rnd.randint(1, 12)
        maxv = rnd.randint(1, 4)        # many collisions
    elif regime == 1:
        n = rnd.randint(1, 40)
        maxv = rnd.randint(1, 8)
    elif regime == 2:
        n = rnd.randint(1, 60)
        maxv = rnd.randint(1, 60)       # mostly distinct
    else:
        n = rnd.randint(1, 30)
        maxv = 1                        # all equal

    q = rnd.randint(1, 30)
    a = [rnd.randint(1, maxv) for _ in range(n)]

    lines = ["%d %d" % (n, q)]
    lines.append(" ".join(map(str, a)))
    for _ in range(q):
        l = rnd.randint(1, n)
        r = rnd.randint(1, n)
        if l > r:
            l, r = r, l
        lines.append("%d %d" % (l, r))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
