import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def reach(A, M):
    """Largest N such that every slot in [0, N] is 'capturable', i.e. lies in
    (A + A) U (A - A)_{>=0}: a rendezvous sum a+b or a phasing difference a-b>=0.
    Returns -1 if slot 0 is not covered (impossible when A is non-empty)."""
    A = sorted(set(A))
    if not A:
        return -1
    mx = A[-1]
    cov = bytearray(2 * mx + 2)          # indices 0 .. 2*mx are reachable at most
    for a in A:
        for b in A:
            cov[a + b] = 1               # rendezvous sum
            d = a - b
            if d >= 0:
                cov[d] = 1               # phasing difference (non-negative)
    N = 0
    while N < len(cov) and cov[N]:
        N += 1
    return N - 1

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    # ---- parse instance ----
    try:
        n = int(inp[0]); M = int(inp[1])
    except Exception:
        fail("bad input")

    # ---- internal baseline B: the arithmetic-progression cordon 0,1,...,n-1 ----
    # Its sum set is exactly [0, 2n-2] (contiguous), so its reach is 2n-2.
    # (Always feasible: M >= n-1.)
    B = max(1, 2 * n - 2)

    # ---- parse participant artifact: k, then k depot slots ----
    try:
        k = int(out[0])
        pos = [int(x) for x in out[1:1 + k]]
    except Exception:
        fail("parse")
    if len(pos) != k:
        fail("count mismatch (declared %d, got %d)" % (k, len(pos)))
    if k < 1 or k > n:
        fail("must place between 1 and %d depots, got %d" % (n, k))

    seen = set()
    for p in pos:
        if p < 0 or p > M:
            fail("depot slot %d out of [0,%d]" % (p, M))
        if p in seen:
            fail("duplicate depot slot %d" % p)
        seen.add(p)

    F = reach(pos, M)
    if F < 0:
        fail("nothing covered")
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("reach=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
