import sys, math

COORD_MAX = 10_000_000

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def isprime(x):
    if x < 2:
        return False
    i = 2
    while i * i <= x:
        if x % i == 0:
            return False
        i += 1
    return True

def next_prime(n):
    p = n
    while not isprime(p):
        p += 1
    return p

def et_ruler(n):
    # Erdos-Turan quadratic-residue Golomb ruler, first n marks.
    p = next_prime(n)
    marks = [2 * p * i + (i * i) % p for i in range(n)]
    return marks

def main():
    try:
        inp = open(sys.argv[1]).read().split()
        n = int(inp[0])
    except Exception:
        fail("bad input")

    if n < 2:
        fail("bad n")

    # ---- internal baseline B (a feasible reference the checker builds itself) ----
    Bmarks = et_ruler(n)
    B = Bmarks[-1] - Bmarks[0]
    B = max(1, B)

    # ---- parse participant output ----
    try:
        toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(toks) != n:
        fail("expected %d integers, got %d" % (n, len(toks)))

    pos = []
    for t in toks:
        try:
            v = float(t)
        except Exception:
            fail("non-numeric token %r" % t)
        if not math.isfinite(v):
            fail("non-finite value")
        if v != math.floor(v):
            fail("non-integer position %r" % t)
        iv = int(v)
        if iv < 0 or iv > COORD_MAX:
            fail("position %d out of range" % iv)
        pos.append(iv)

    # distinct positions
    if len(set(pos)) != n:
        fail("duplicate antenna positions")

    # all pairwise separations distinct (non-redundant / Golomb condition)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(pos[i] - pos[j])
            if d == 0 or d in diffs:
                fail("redundant baseline: separation %d repeats" % d)
            diffs.add(d)

    F = max(pos) - min(pos)
    if F <= 0:
        fail("degenerate span")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print("span=%d baseline=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
