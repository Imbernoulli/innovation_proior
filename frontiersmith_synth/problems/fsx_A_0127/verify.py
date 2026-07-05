import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def total_service(A):
    """F = |A+A| + |A-A|, computed exactly."""
    m = len(A)
    sums = set()
    diffs = set()
    for i in range(m):
        ai = A[i]
        for j in range(i, m):
            aj = A[j]
            sums.add(ai + aj)
            d = ai - aj
            diffs.add(d)
            diffs.add(-d)
    return len(sums) + len(diffs)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        k = int(inp[0])
        V = int(inp[1])
    except Exception:
        fail("bad input")

    # ---- internal baseline B: contiguous ladder {0,...,m0-1} ----
    m0 = min(k, V + 1)
    B = 4 * m0 - 2
    B = max(1, B)

    # ---- parse participant output ----
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not out:
        fail("empty output")
    try:
        m = int(out[0])
        vals = out[1:1 + m]
    except Exception:
        fail("parse")
    if m < 0 or m > k:
        fail("count %d out of [0,%d]" % (m, k))
    if len(vals) != m:
        fail("count mismatch: header %d but %d numbers" % (m, len(vals)))

    A = []
    seen = set()
    for s in vals:
        try:
            x = int(s)
        except Exception:
            fail("non-integer milepost %r" % s)
        if x < 0 or x > V:
            fail("milepost %d out of [0,%d]" % (x, V))
        if x in seen:
            fail("duplicate milepost %d" % x)
        seen.add(x)
        A.append(x)

    F = total_service(A)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
