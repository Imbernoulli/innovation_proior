#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE power-grid instance to stdout.

Difficulty ladder is a growing number of transmission lines n (state space 3^n).
Each instance also carries a deterministic set of RESERVED substation
configurations (blocked vectors) that may not be deployed.  Reserved configs
always carry phase 2 on the last line, so they never touch the audit baseline
(which lives on phase-0 last line) -- this keeps the normaliser stable while
still perturbing every richer strategy per test.
"""
import sys, random

# testId 1..10 -> number of transmission lines n (small -> large)
LADDER = [3, 4, 4, 5, 5, 6, 6, 7, 7, 7]


def main():
    tid = int(sys.argv[1])
    n = LADDER[(tid - 1) % len(LADDER)]
    rng = random.Random(9_000 + tid)
    k = n  # number of reserved configurations
    blocked = set()
    while len(blocked) < k:
        v = tuple(rng.randint(0, 2) for _ in range(n - 1)) + (2,)
        blocked.add(v)
    blocked = sorted(blocked)
    out = ["%d %d" % (n, k)]
    for v in blocked:
        out.append(" ".join(map(str, v)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
