import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def channels(A):
    """Return |A+A| + |A-A| exactly."""
    sums = set()
    diffs = set()
    for a in A:
        for b in A:
            sums.add(a + b)
            diffs.add(a - b)
    return len(sums) + len(diffs)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    # ---- parse instance ----
    try:
        n = int(inp[0]); M = int(inp[1])
    except Exception:
        fail("bad input")

    # ---- internal baseline B: arithmetic run {0,1,...,n-1} (always fits, M >= n-1) ----
    A0 = list(range(n))
    B = channels(A0)          # = 4n - 2 for an arithmetic run
    B = max(1, B)

    # ---- parse participant artifact: k, then k slots ----
    try:
        k = int(out[0])
        pos = [int(x) for x in out[1:1 + k]]
    except Exception:
        fail("parse")
    if len(pos) != k:
        fail("count mismatch (declared %d, got %d)" % (k, len(pos)))
    if k != n:
        fail("need exactly %d slots, got %d" % (n, k))

    seen = set()
    for p in pos:
        if p < 0 or p > M:
            fail("slot %d out of [0,%d]" % (p, M))
        if p in seen:
            fail("duplicate slot %d" % p)
        seen.add(p)

    F = channels(pos)          # |A+A| + |A-A|
    sc = min(1000.0, 100.0 * F / B)
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
