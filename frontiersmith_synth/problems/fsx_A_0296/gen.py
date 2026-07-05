#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE subway-polarity instance to stdout.

testId 1..8 is a difficulty ladder over the station count N (odd, 13..27, so no
Hadamard matrix of that order exists -> the optimum stays genuinely unknown).

Instance format (stdout):
    line 1:  N seed K
    next K:  i j v      # fixed track polarity: entry (i,j) MUST equal v in {-1,+1}
                        # (0-indexed). The N diagonal cells are always fixed to +1.

Everything is deterministic in testId only.
"""
import sys


class LCG:
    """Platform-independent 64-bit LCG (identical logic in gen/verify/solutions)."""
    __slots__ = ("x",)

    def __init__(self, seed):
        self.x = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)

    def nxt(self):
        self.x = (self.x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.x

    def bit(self):
        return 1 if (self.nxt() >> 33) & 1 else -1

    def rint(self, m):
        return self.nxt() % m


def make_fixed(n, seed):
    """Deterministic set of fixed cells: all diagonal (+1) plus K off-diagonal."""
    rng = LCG(seed ^ 0x5BD1E995)
    fixed = {}
    for i in range(n):
        fixed[(i, i)] = 1
    K = n  # about one extra fixed track per station
    cnt = 0
    guard = 0
    while cnt < K and guard < 40 * K:
        guard += 1
        i = rng.rint(n)
        j = rng.rint(n)
        if i == j or (i, j) in fixed:
            continue
        fixed[(i, j)] = rng.bit()
        cnt += 1
    return fixed


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid < 1:
        tid = 1
    n = 11 + 2 * tid          # 13,15,...,27 (all odd)
    seed = 1000003 * tid + 20260701
    fixed = make_fixed(n, seed)
    items = sorted(fixed.items())
    out = ["%d %d %d" % (n, seed, len(items))]
    for (i, j), v in items:
        out.append("%d %d %d" % (i, j, v))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
