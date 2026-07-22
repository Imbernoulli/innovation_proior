# TIER: greedy
# The "obvious" textbook move: single-pass pairwise common-subexpression
# elimination (CSE).  Score every pair of input columns by how many rows use
# BOTH of them, fold the most-reused pairs into shared XOR lines (one static
# pass, no iterative re-scoring / no chaining of shared terms into deeper
# shared terms), then finish each row independently on whatever bits remain.
# This is real sharing and clearly beats doing nothing -- but it only ever
# folds RAW COLUMN pairs, so it cannot discover the multi-way (~m/2-wide)
# planted correction that "strong" exploits; it is blind to that structure.
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

    # column bitmask over ROWS: col_mask[j] has bit i set iff column j is 1 in row i
    col_mask = [0] * m
    for i, bm in enumerate(rows):
        b = bm
        while b:
            low = b & (-b)
            j = low.bit_length() - 1
            col_mask[j] |= (1 << i)
            b ^= low

    # candidate pairs with static co-occurrence frequency >= 2
    cand = []
    for j in range(m):
        cj = col_mask[j]
        if cj == 0:
            continue
        for k in range(j + 1, m):
            freq = bin(cj & col_mask[k]).count("1")
            if freq >= 2:
                cand.append((freq, j, k))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))

    ops = []
    next_line = m + 1
    remaining = list(rows)                 # bits not yet folded, per row
    row_terms = [[] for _ in range(m)]     # extra shared-term line ids folded in, per row
    pair_line = {}

    for freq, j, k in cand:
        pat = (1 << j) | (1 << k)
        rows_here = [i for i in range(m) if (remaining[i] & pat) == pat]
        if len(rows_here) < 2:
            continue
        ops.append((j + 1, k + 1))
        tline = next_line
        next_line += 1
        pair_line[(j, k)] = tline
        for i in rows_here:
            remaining[i] &= ~pat
            row_terms[i].append(tline)

    outrefs = [0] * m
    for i in range(m):
        terms = list(row_terms[i])
        b = remaining[i]
        while b:
            low = b & (-b)
            jcol = low.bit_length() - 1
            terms.append(jcol + 1)  # 1-indexed input line
            b ^= low
        terms.sort()
        if not terms:
            outrefs[i] = 1  # unreachable (defensive; rows are never all-zero)
            continue
        cur = terms[0]
        for tval in terms[1:]:
            ops.append((cur, tval))
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
