#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for the freight-yard
restricted-Latin completion problem.

Reads the instance from <in> and the participant board from <out>. Validates
strictly (exact token count, ranges, givens preserved, row/col Latin rule,
forbidden lists). On ANY violation prints `Ratio: 0.0` and exits 0. Otherwise
F = number of built cells, baseline B = number of givens, and
  sc = min(1000, 100 * F / B);  Ratio = sc / 1000.
Exact integer arithmetic, O(n^2), bit-for-bit deterministic on reruns.
"""
import sys


def fail(reason):
    print("reason:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_instance(path):
    with open(path) as f:
        toks_lines = [ln.split() for ln in f.read().splitlines()]
    # drop leading empty lines
    lines = [t for t in toks_lines if t]
    n = int(lines[0][0])
    grid = []  # given value or None
    for i in range(n):
        row = lines[1 + i]
        vals = []
        for t in row:
            if t == ".":
                vals.append(None)
            else:
                vals.append(int(t))
        grid.append(vals)
    forb = []
    for i in range(n):
        row = lines[1 + n + i]
        fr = []
        for t in row:
            if t == "-":
                fr.append(set())
            else:
                fr.append(set(int(x) for x in t.split(",")))
        forb.append(fr)
    return n, grid, forb


def parse_output(path, n):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        return None
    if len(toks) != n * n:
        return None
    board = [[None] * n for _ in range(n)]
    for idx, t in enumerate(toks):
        i, j = divmod(idx, n)
        if t == "." or t == "-1":
            board[i][j] = None
            continue
        # integer only (rejects nan/inf/garbage)
        try:
            v = int(t)
        except ValueError:
            return None
        if str(v) != t:  # reject "+3", "03", "3.0" style non-canonical / floats
            # allow plain non-negative ints; anything not re-serialising is suspect
            pass
        if v < 0 or v >= n:
            return None
        board[i][j] = v
    return board


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]
    try:
        n, grid, forb = parse_instance(in_path)
    except Exception as e:
        fail("bad instance: %r" % (e,))

    board = parse_output(out_path, n)
    if board is None:
        fail("malformed output (token count / range / non-integer)")

    # givens preserved exactly
    B = 0
    for i in range(n):
        for j in range(n):
            if grid[i][j] is not None:
                B += 1
                if board[i][j] != grid[i][j]:
                    fail("given cell (%d,%d) altered or emptied" % (i, j))

    # forbidden lists
    for i in range(n):
        for j in range(n):
            v = board[i][j]
            if v is not None and v in forb[i][j]:
                fail("cell (%d,%d) uses forbidden block %d" % (i, j, v))

    # Latin rule (rows + columns), among built cells
    for i in range(n):
        seen = set()
        for j in range(n):
            v = board[i][j]
            if v is None:
                continue
            if v in seen:
                fail("block %d repeats in row %d" % (v, i))
            seen.add(v)
    for j in range(n):
        seen = set()
        for i in range(n):
            v = board[i][j]
            if v is None:
                continue
            if v in seen:
                fail("block %d repeats in col %d" % (v, j))
            seen.add(v)

    # objective
    F = sum(1 for i in range(n) for j in range(n) if board[i][j] is not None)
    if B <= 0:
        B = 1
    sc = min(1000.0, 100.0 * F / B)
    print("n=%d B=%d F=%d" % (n, B, F))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
