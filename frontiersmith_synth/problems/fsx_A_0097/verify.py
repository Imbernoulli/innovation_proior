import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        k = int(inp[0])
        V = int(inp[1])
    except Exception:
        fail("bad input")

    # ---- internal baseline B: contiguous ramp {0,...,m0-1} ----
    m0 = min(k, V + 1)
    B = 2 * m0 - 1
    B = max(1, B)

    # ---- parse participant output ----
    out = open(sys.argv[2]).read().split()
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
            fail("non-integer offset %r" % s)
        if x < 0 or x > V:
            fail("offset %d out of [0,%d]" % (x, V))
        if x in seen:
            fail("duplicate offset %d" % x)
        seen.add(x)
        A.append(x)

    # ---- exact sumset |A+A| ----
    sums = set()
    for i in range(m):
        ai = A[i]
        for j in range(i, m):
            sums.add(ai + A[j])
    F = len(sums)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
