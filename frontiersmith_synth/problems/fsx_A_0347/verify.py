import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    # ---- read instance ----
    try:
        inp = open(sys.argv[1]).read().split()
        k = int(inp[0])
        M = int(inp[1])
    except Exception:
        fail("bad instance")

    # ---- internal trivial baseline B: the length-k arithmetic progression ----
    # AP = {0,1,...,k-1} (fits since M = 9k > k-1); |AP+AP| = 2k-1 exactly.
    B = max(1, 2 * k - 1)

    # ---- read participant artifact (bounded, strict) ----
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(raw) < 1:
        fail("empty output")

    # first token = declared count c
    try:
        c = int(raw[0])
    except Exception:
        fail("count not an integer")

    if c < 1 or c > k:
        fail("count out of range [1,%d]" % k)

    # exactly c element tokens must follow -- no more, no less
    if len(raw) != 1 + c:
        fail("token count %d != 1+c (%d)" % (len(raw), 1 + c))

    A = []
    seen = set()
    for t in raw[1:1 + c]:
        # int() rejects nan / inf / floats / garbage automatically
        try:
            v = int(t)
        except Exception:
            fail("non-integer offset %r" % t)
        if v < 0 or v > M:
            fail("offset %d outside [0,%d]" % (v, M))
        if v in seen:
            fail("duplicate offset %d" % v)
        seen.add(v)
        A.append(v)

    if len(A) != c:
        fail("internal count mismatch")

    # ---- objective: exact sumset cardinality |A+A| ----
    sums = set()
    for x in A:
        for y in A:
            sums.add(x + y)
    F = len(sums)

    # maximization normalization (matches AGENT_BRIEF): trivial ~= 0.1, cap at 1.0
    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
