import sys

MAX_M = 5000
MAX_K = 500
MAX_E = 10 ** 7
MAX_TOKENS = 2_000_000


def fail(reason):
    print("INFEASIBLE: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints(path, limit):
    with open(path, "r") as f:
        data = f.read()
    toks = data.split()
    if len(toks) > limit:
        return None
    out = []
    for t in toks:
        try:
            out.append(int(t))
        except ValueError:
            return None
    return out


def main():
    if len(sys.argv) < 3:
        fail("bad checker invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    itoks = read_ints(in_path, MAX_TOKENS)
    if itoks is None or len(itoks) < 4:
        fail("cannot parse input file")
    pos = 0
    p = itoks[pos]; pos += 1
    g = itoks[pos]; pos += 1
    LAMBDA = itoks[pos]; pos += 1
    T = itoks[pos]; pos += 1
    if p < 2 or T < 0 or len(itoks) < pos + T:
        fail("input file corrupt")
    targets = itoks[pos:pos + T]

    otoks = read_ints(out_path, MAX_TOKENS)
    if otoks is None:
        fail("cannot parse participant output (non-integer token, or too large)")
    if len(otoks) < 1:
        fail("empty output")

    j = 0

    def nxt():
        nonlocal j
        if j >= len(otoks):
            return None
        v = otoks[j]
        j += 1
        return v

    m = nxt()
    if m is None or m < 1 or m > MAX_M:
        fail("m out of range [1,%d]" % MAX_M)

    table = []
    for _ in range(m):
        s = nxt()
        if s is None or s < 1 or s > p - 1:
            fail("table entry out of range [1,p-1]")
        table.append(s)

    def modpow_signed(base, e, mod):
        if e >= 0:
            return pow(base, e, mod)
        inv = pow(base, mod - 2, mod)
        return pow(inv, -e, mod)

    total_factors = 0
    for ti in range(T):
        k = nxt()
        if k is None or k < 1 or k > MAX_K:
            fail("k_%d out of range [1,%d]" % (ti + 1, MAX_K))
        prod = 1
        for _ in range(k):
            idx = nxt()
            e = nxt()
            if idx is None or e is None:
                fail("truncated factor list for target %d" % (ti + 1))
            if idx < 1 or idx > m:
                fail("index out of range for target %d" % (ti + 1))
            if e < -MAX_E or e > MAX_E:
                fail("exponent out of range for target %d" % (ti + 1))
            base = table[idx - 1]
            prod = (prod * modpow_signed(base, e, p)) % p
        if prod != targets[ti] % p:
            fail("assembled product mismatches target %d" % (ti + 1))
        total_factors += k

    if j != len(otoks):
        # extra trailing garbage beyond what was needed
        fail("unexpected trailing tokens in output")

    F = total_factors + LAMBDA * m
    B = T * (1 + LAMBDA)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("total_factors=%d m=%d F=%d B=%d" % (total_factors, m, F, B))
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
