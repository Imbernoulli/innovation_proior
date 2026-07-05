import sys


def fail(reason):
    # Feasibility violation -> zero score.
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    return int(toks[0])


def parse_output(path, n):
    with open(path) as f:
        raw = f.read().split()
    # Bounded: a feasible set has at most n distinct values in [1,n].
    if len(raw) > n + 5:
        fail("too many tokens")
    vals = []
    for tok in raw:
        # Strict integer parse; rejects nan/inf/floats/exponent forms.
        try:
            v = int(tok)
        except (ValueError, TypeError):
            fail("non-integer token %r" % tok)
        vals.append(v)
    return vals


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    n = read_instance(in_path)

    vals = parse_output(out_path, n)

    # Empty set is feasible but scores 0.
    if len(vals) == 0:
        print("empty set")
        print("Ratio: 0.0")
        sys.exit(0)

    # Range check.
    for v in vals:
        if v < 1 or v > n:
            fail("value %d out of range [1,%d]" % (v, n))

    # Distinctness.
    S = set(vals)
    if len(S) != len(vals):
        fail("duplicate codewords")

    # Sidon / conflict-free check: all pairwise sums (a<=b) distinct.
    arr = sorted(S)
    sums = set()
    k = len(arr)
    for i in range(k):
        ai = arr[i]
        for j in range(i, k):
            s = ai + arr[j]
            if s in sums:
                fail("pairwise-sum collision at %d" % s)
            sums.add(s)

    F = float(len(arr))

    # Internal baseline B: powers of two <= n (always a conflict-free code).
    B = 0
    x = 1
    while x <= n:
        B += 1
        x *= 2
    B = float(B)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d n=%d" % (int(F), int(B), n))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
