#!/usr/bin/env python3
# Random small-case generator for Cut-to-Quarantine.
# Usage: python3 gen.py <seed>
# Emits a valid instance on stdout:
#   n
#   (n-1) lines: u v w     (tree edges, 1<=w<=W)
#   q
#   q queries: k  s1 s2 ... sk   (special nodes, distinct, never node 1)

import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 12)
    W = rng.choice([1, 5, 20, 1000])  # vary the weight magnitude

    lines = []
    lines.append(str(n))
    for v in range(2, n + 1):
        # attach v to a random earlier node -> guarantees a tree
        p = rng.randint(1, v - 1)
        w = rng.randint(1, W)
        # randomize endpoint order to avoid any directional assumption
        if rng.random() < 0.5:
            lines.append(f"{p} {v} {w}")
        else:
            lines.append(f"{v} {p} {w}")

    q = rng.randint(1, 6)
    lines.append(str(q))
    candidates = list(range(2, n + 1))  # node 1 (root) is never special
    for _ in range(q):
        if not candidates:
            lines.append("0")
            continue
        k = rng.randint(0, len(candidates))
        if k == 0:
            lines.append("0")
            continue
        spec = rng.sample(candidates, k)
        lines.append(f"{k} " + " ".join(map(str, spec)))

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
