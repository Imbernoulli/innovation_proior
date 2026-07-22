import sys

# Difficulty ladder: (R, C) -- a frame-bordered corridor.
# Row 0, row R-1, col 0, col C-1 are permanent KEEP material ('.').
# Every interior cell (rows 1..R-2, cols 1..C-2) is WASTE ('#') that must be
# removed exactly once. Larger C = a longer corridor the clamps must travel
# along; larger R = more depth (more waste rows) to clear at each column.
LADDER = {
    1:  (3, 8),
    2:  (4, 16),
    3:  (4, 22),
    4:  (5, 28),
    5:  (5, 32),
    6:  (5, 38),
    7:  (6, 42),
    8:  (6, 46),
    9:  (6, 50),
    10: (6, 56),
}

PEN = 0.005  # refixture penalty per reposition move (same units as margin)


def build_grid(R, C):
    rows = []
    for r in range(R):
        if r == 0 or r == R - 1:
            rows.append('.' * C)
        else:
            row = ['.'] + ['#'] * (C - 2) + ['.']
            rows.append(''.join(row))
    return rows


def main():
    i = int(sys.argv[1])
    R, C = LADDER.get(i, LADDER[1])
    K = 4
    rows = build_grid(R, C)
    clamps = [(0, 0), (0, 2), (R - 1, 0), (R - 1, 2)]

    out = []
    out.append("%d %d %d %.6f" % (R, C, K, PEN))
    out.extend(rows)
    for (r, c) in clamps:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
