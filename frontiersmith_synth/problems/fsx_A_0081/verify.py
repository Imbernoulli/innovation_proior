import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    try:
        it = iter(inp)
        n = int(next(it))
        b = int(next(it))
        flooded = set()
        for _ in range(b):
            flooded.add(next(it))
    except Exception:
        fail("bad input")

    valid_chars = set("012")

    # ---- internal baseline B: ring diagonal restricted to non-flooded ----
    diag = ["0" * n] + ["".join("1" if j == i else "0" for j in range(n)) for i in range(n)]
    B = sum(1 for c in diag if c not in flooded)
    B = max(1, B)

    # ---- parse participant output ----
    try:
        k = int(out[0])
        cells = out[1:1 + k]
    except Exception:
        fail("parse")
    if len(cells) != k:
        fail("count mismatch")

    seen = set()
    for c in cells:
        if len(c) != n or any(ch not in valid_chars for ch in c):
            fail("bad address %r" % c)
        if c in seen:
            fail("duplicate %s" % c)
        if c in flooded:
            fail("flooded %s" % c)
        seen.add(c)

    # ---- raiding-line (cap set) check: for each pair, third completer must be absent ----
    arr = cells
    m = len(arr)
    # decode to tuples once
    tup = [tuple(ord(ch) - 48 for ch in c) for c in arr]
    sset = seen
    for a in range(m):
        ta = tup[a]
        for bb in range(a + 1, m):
            tb = tup[bb]
            third = "".join(chr(48 + ((3 - (ta[i] + tb[i]) % 3) % 3)) for i in range(n))
            if third in sset and third != arr[a] and third != arr[bb]:
                fail("raiding line %s %s %s" % (arr[a], arr[bb], third))

    F = k
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
