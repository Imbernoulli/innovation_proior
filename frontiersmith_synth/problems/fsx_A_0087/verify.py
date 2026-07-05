import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def diff_set_size(A):
    s = set()
    for a in A:
        for b in A:
            s.add(a - b)
    return len(s)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    # ---- parse instance ----
    try:
        n = int(inp[0]); M = int(inp[1])
    except Exception:
        fail("bad input")

    # ---- internal baseline B: arithmetic progression 0..n-1 (always feasible: M >= n-1) ----
    # |A-A| of an AP of size n is exactly 2n-1.
    B = max(1, 2 * n - 1)

    # ---- parse participant artifact: k, then k positions ----
    try:
        k = int(out[0])
        pos = [int(x) for x in out[1:1 + k]]
    except Exception:
        fail("parse")
    if len(pos) != k:
        fail("count mismatch (declared %d, got %d)" % (k, len(pos)))
    if k != n:
        fail("need exactly %d stages, got %d" % (n, k))

    seen = set()
    for p in pos:
        if p < 0 or p > M:
            fail("position %d out of [0,%d]" % (p, M))
        if p in seen:
            fail("duplicate position %d" % p)
        seen.add(p)

    F = diff_set_size(pos)   # |A-A|, the number of distinct signed spacings (incl. 0)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
