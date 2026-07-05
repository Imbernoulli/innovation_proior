#!/usr/bin/env python3
"""Instance generator for fsx_C_0396 - Traffic Signal Phase Grid completion.

Usage: python3 gen.py <testId>
Prints ONE instance to stdout. testId in 1..N drives a difficulty ladder
(small -> large). All randomness is seeded ONLY by testId, so instances are
bit-for-bit reproducible.

Instance = a partially specified N x N grid of traffic intersections.  Each
prefilled cell holds a signal PHASE symbol in 0..N-1.  The prefilled cells form
a valid partial Latin square (no phase repeats within a row/column), generated
by seeded random valid placement up to a target density.
"""
import sys


class LCG:
    """Deterministic linear congruential generator (no library RNG)."""
    def __init__(self, seed):
        self.s = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (self.s >> 17) & 0x7FFFFFFF

    def randint(self, n):
        return self.next() % n

    def shuffle(self, arr):
        for i in range(len(arr) - 1, 0, -1):
            j = self.randint(i + 1)
            arr[i], arr[j] = arr[j], arr[i]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    if t < 1:
        t = 1

    N = 16 + 3 * t                # ladder: t=1 -> 19, t=10 -> 46
    density = 0.33                # fraction of cells prefilled
    K = int(density * N * N)

    rng = LCG(0x9E3779B1 * t + 12345)

    grid = [[-1] * N for _ in range(N)]
    row_used = [0] * N            # bitmask of phases used in each row
    col_used = [0] * N            # bitmask of phases used in each column
    full = (1 << N) - 1

    # random cell visitation order
    cells = [(i, j) for i in range(N) for j in range(N)]
    rng.shuffle(cells)

    placed = 0
    for (i, j) in cells:
        if placed >= K:
            break
        avail = full & ~row_used[i] & ~col_used[j]
        if avail == 0:
            continue
        # collect available phases
        options = []
        m = avail
        while m:
            b = m & (-m)
            options.append(b.bit_length() - 1)
            m ^= b
        s = options[rng.randint(len(options))]
        grid[i][j] = s
        row_used[i] |= (1 << s)
        col_used[j] |= (1 << s)
        placed += 1

    out = [str(N)]
    for i in range(N):
        out.append(" ".join(str(x) for x in grid[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
