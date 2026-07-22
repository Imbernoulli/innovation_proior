import sys

# Format D checker -- custom-alphabet-order + rotation BWT run-count minimization.
#   1) Parse instance: n, k, T[0..n-1] (symbols in [0,k-1]) from <in>.
#   2) Parse participant output from <out>: k+1 whitespace tokens = a permutation p of
#      0..k-1 (p[i] = symbol with rank i) followed by a rotation r in [0,n).
#   3) Feasibility gate: exact token count, valid permutation, valid rotation, all
#      integers (non-finite / non-integer tokens fail to parse -> reject).
#   4) Objective (minimize) = runs(p, r): rotate T by r, append a sentinel that always
#      ranks below every real symbol, form the circular suffix array under the order
#      induced by p, read off the BWT (last) column, count maximal equal-symbol runs.
#   5) Baseline B = n+1 -- the trivial, always-valid upper bound (a length-(n+1) BWT
#      string trivially cannot have more than n+1 maximal runs; no construction needed).
#      Ratio = min(1, 0.1*B/F), printed as min(1000, 100*B/F)/1000.


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def circular_suffix_array(rank_arr, m):
    """Sort the m cyclic rotations of an m-length rank-encoded string.  Standard
    prefix-doubling algorithm on a circular index space; O(m log^2 m)."""
    rank = rank_arr[:]
    sa = list(range(m))
    shift = 1
    while True:
        def key(i):
            return (rank[i], rank[(i + shift) % m])
        sa.sort(key=key)
        tmp = [0] * m
        tmp[sa[0]] = 0
        for idx in range(1, m):
            tmp[sa[idx]] = tmp[sa[idx - 1]] + (1 if key(sa[idx - 1]) < key(sa[idx]) else 0)
        rank = tmp
        if rank[sa[-1]] == m - 1:
            break
        shift <<= 1
        if shift > m:
            break
    return sa


def bwt_run_count(seq, n, k, order, r):
    """order: permutation of 0..k-1, order[i] = symbol with rank i.  r: rotation offset.
    Returns the number of maximal equal-symbol runs in the resulting BWT column."""
    rank_of = [0] * k
    for i, s in enumerate(order):
        rank_of[s] = i
    m = n + 1
    rank_arr = [0] * m       # rank-encoding used for sorting (sentinel = 0, symbols 1..k)
    sym = [0] * m            # actual symbol identity used for run-counting (sentinel = -1)
    for i in range(n):
        s = seq[(r + i) % n]
        rank_arr[i] = rank_of[s] + 1
        sym[i] = s
    rank_arr[n] = 0
    sym[n] = -1
    sa = circular_suffix_array(rank_arr, m)
    runs = 0
    prev = None
    for idx in sa:
        ch = sym[(idx - 1) % m]
        if ch != prev:
            runs += 1
            prev = ch
    return runs


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input file")
    it = iter(inp)
    try:
        n = int(next(it))
        k = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= n <= 2000 and 2 <= k <= 200):
        fail("dims out of range")
    seq = []
    try:
        for _ in range(n):
            v = int(next(it))
            seq.append(v)
    except Exception:
        fail("bad/short sequence")
    if any(v < 0 or v >= k for v in seq):
        fail("symbol out of range")

    try:
        out_tokens = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output file")

    if len(out_tokens) != k + 1:
        fail("expected %d tokens (k order + 1 rotation), got %d" % (k + 1, len(out_tokens)))

    try:
        order = [int(t) for t in out_tokens[:k]]
        r = int(out_tokens[k])
    except Exception:
        fail("non-integer token (nan/inf/float/garbage rejected)")

    if sorted(order) != list(range(k)):
        fail("submitted order is not a permutation of 0..k-1")
    if not (0 <= r < n):
        fail("rotation r out of range")

    F = bwt_run_count(seq, n, k, order, r)
    B = n + 1  # trivial upper bound: a length-(n+1) column has at most n+1 runs

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("n=%d k=%d F=%d B=%d Ratio: %.6f" % (n, k, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
