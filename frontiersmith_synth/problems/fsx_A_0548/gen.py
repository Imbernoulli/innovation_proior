#!/usr/bin/env python3
# gen.py <testId>  ->  prints ONE instance of a target m x n GF(2) matrix to stdout.
#
# Difficulty ladder testId 1..10 (small -> large). Every case plants the SAME structure:
# each target row is the complement of a SPARSE random set of size k (so rows are DENSE,
# weight n-k > n/2).  This is the cancellation trap:
#   * independent per-row folds cost ~ (n-k-1) each (baseline);
#   * monotone common-subexpression sharing helps only within each row's support, and no
#     shared intermediate may contain a bit missing from a row that uses it -- so it cannot
#     build the all-ones sum;
#   * the cancellation insight builds S = x_0 ^ ... ^ x_{n-1} ONCE, then realises each dense
#     row as S ^ (fold of its k MISSING bits) at cost ~ k, unreachable by cancellation-free
#     sharing.
# A shared "anchor" block of columns is also planted so monotone sharing beats the baseline
# (greedy > trivial) while cancellation beats greedy.
import sys, random

# (n, m, k) per testId.  n>2k+1 keeps every row dense enough that cancellation wins.
LADDER = {
    1:  (24, 22, 3),
    2:  (26, 24, 3),
    3:  (28, 26, 4),
    4:  (30, 28, 4),
    5:  (32, 30, 4),
    6:  (34, 32, 4),
    7:  (36, 34, 5),
    8:  (38, 36, 5),
    9:  (40, 38, 5),
    10: (40, 40, 6),
}

def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid not in LADDER:
        tid = ((tid - 1) % 10) + 1
    n, m, k = LADDER[tid]
    rng = random.Random(700_000 + tid)

    cols = list(range(n))
    # A small planted "anchor" set of columns that is missing from NONE of the rows,
    # giving monotone sharing a common sub-sum to exploit (so greedy clears trivial).
    anchor = set(rng.sample(cols, max(2, n // 6)))
    missable = [c for c in cols if c not in anchor]

    full = (1 << n) - 1
    rows = []
    seen = set()
    guard = 0
    while len(rows) < m:
        guard += 1
        miss = rng.sample(missable, k)          # the k MISSING bits (never from anchor)
        mm = 0
        for c in miss:
            mm |= (1 << c)
        row = full ^ mm                         # dense: weight n-k
        if row == 0:
            continue
        if row in seen and guard < 20 * m:      # prefer distinct rows, but don't loop forever
            continue
        seen.add(row)
        rows.append(row)

    out = ["%d %d" % (m, n)]
    for r in rows:
        out.append(" ".join("1" if (r >> j) & 1 else "0" for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
