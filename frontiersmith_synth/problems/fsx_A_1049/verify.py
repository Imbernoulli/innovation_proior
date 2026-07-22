import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    try:
        in_lines = [ln.rstrip("\n\r") for ln in open(sys.argv[1]).readlines()]
    except Exception:
        fail("cannot read input")

    try:
        head = in_lines[0].split()
        L = int(head[0]); k = int(head[1]); d = int(head[2]); bonus = float(head[3])
        letters = []
        pos = 1
        for _ in range(L):
            rows = in_lines[pos:pos + 7]
            pos += 7
            bit = "".join(r.strip() for r in rows)
            if len(bit) != 35 or any(c not in "01" for c in bit):
                fail("corrupt instance")
            letters.append(bit)
        if len(set(letters)) != L:
            fail("corrupt instance (duplicate letters)")
    except Exception:
        fail("bad input")

    def decode(bitmap):
        # returns (letter_index, margin) or None if infeasible (not within d
        # of the dictionary, or the nearest glyph is not unique)
        if len(bitmap) != 35 or any(c not in "01" for c in bitmap):
            return None
        dists = [hamming(bitmap, w) for w in letters]
        s = sorted(dists)
        if s[0] > d or s[0] >= s[1]:
            return None
        return dists.index(s[0]), s[1] - s[0]

    # ---- internal baseline B: repeat the first self-compatible (palindrome)
    # dictionary letter in every slot (always feasible by construction) ----
    anchor = None
    for i, w in enumerate(letters):
        if w == w[::-1]:
            anchor = i
            break
    if anchor is None:
        fail("instance has no self-compatible letter (generator bug)")
    dec = decode(letters[anchor])
    anchor_margin = dec[1] if dec is not None else 0.0
    B = max(1e-9, 2.0 * k * anchor_margin + bonus)

    # ---- parse participant grid: 7 lines of length 5k, chars in {0,1} ----
    try:
        raw = open(sys.argv[2]).read()
    except Exception:
        fail("cannot read output")
    # Strict row count: only tolerate a trailing '\r' (CRLF); do NOT drop
    # blank lines (leading, interior, or trailing) or strip interior
    # whitespace -- the statement requires exactly 7 rows of exactly 5k
    # chars each, full stop.
    lines = [ln.rstrip("\r") for ln in raw.splitlines()]
    if len(lines) != 7:
        fail("expected exactly 7 grid rows, got %d" % len(lines))
    W = 5 * k
    for ln in lines:
        if len(ln) != W or any(c not in "01" for c in ln):
            fail("bad grid row (need length %d, chars in {0,1})" % W)
    grid = lines

    # 180-degree rotation of the whole grid: reverse row order + reverse
    # each row's character order.
    rgrid = [grid[6 - r][::-1] for r in range(7)]

    def slot(rows, j):
        # j is 1-indexed slot number; concatenate the 7x5 block row-major
        return "".join(rows[r][5 * (j - 1):5 * j] for r in range(7))

    idx_a = [None] * (k + 1)
    idx_b = [None] * (k + 1)
    margin_sum = 0.0
    for j in range(1, k + 1):
        du = decode(slot(grid, j))
        if du is None:
            fail("slot %d does not decode upright within distance %d" % (j, d))
        db = decode(slot(rgrid, j))
        if db is None:
            fail("slot %d does not decode after 180-degree rotation" % j)
        idx_a[j], mu = du
        idx_b[j], mb = db
        margin_sum += mu + mb

    center = (k + 1) // 2
    if idx_a[center] != idx_b[center]:
        fail("center slot %d is not a rotation near-fixed-point (upright=%s, rotated=%s)"
             % (center, idx_a[center], idx_b[center]))

    distinct = set(idx_a[1:k + 1]) | set(idx_b[1:k + 1])
    F = margin_sum + bonus * len(distinct)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f distinct=%d Ratio: %.6f" % (F, B, len(distinct), sc / 1000.0))


if __name__ == "__main__":
    main()
