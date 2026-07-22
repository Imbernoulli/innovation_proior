# TIER: trivial
# Reproduce the checker's own baseline B exactly: for each output row,
# independently chain-XOR its own set input bits (increasing column order).
# No sharing across rows at all -> op count == B -> Ratio == 0.1 exactly.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it))
    rows = []
    for _ in range(m):
        s = next(it)
        bm = 0
        for j, c in enumerate(s):
            if c == "1":
                bm |= (1 << j)
        rows.append(bm)

    ops = []          # list of (a, b) 1-indexed line refs
    next_line = m + 1
    outrefs = [0] * m

    for i, bm in enumerate(rows):
        cols = [j for j in range(m) if (bm >> j) & 1]
        if not cols:
            outrefs[i] = 1  # unreachable given problem guarantees (defensive)
            continue
        cur = cols[0] + 1  # 1-indexed input line
        for c in cols[1:]:
            ops.append((cur, c + 1))
            cur = next_line
            next_line += 1
        outrefs[i] = cur

    out = [str(len(ops))]
    for (a, b) in ops:
        out.append("%d %d" % (a, b))
    out.append(" ".join(map(str, outrefs)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
