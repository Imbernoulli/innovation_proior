import sys

MAX_WORDS = 5000  # honest optima are ~50; anything larger is infeasible.

def popcount(x):
    return bin(x).count("1")

def fail(msg):
    print("%s Ratio: 0.0" % msg)
    sys.exit(0)

def main():
    if len(sys.argv) < 3:
        fail("usage.")
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        parts = f.read().split()
    n, w, d = int(parts[0]), int(parts[1]), int(parts[2])

    try:
        with open(outf) as f:
            data = f.read().split()
    except Exception:
        fail("no output.")

    if len(data) > MAX_WORDS:
        fail("too many barcodes.")

    tmax = w - d // 2  # allowed overlap; distance = 2w - 2*overlap >= d
    if tmax < 0:
        tmax = 0

    masks = []
    seen = set()
    for tok in data:
        if len(tok) != n:
            fail("bad length.")
        m = 0
        ones = 0
        ok = True
        for i, c in enumerate(tok):
            if c == '1':
                m |= (1 << i)
                ones += 1
            elif c != '0':
                ok = False
                break
        if not ok:
            fail("non-binary token.")
        if ones != w:
            fail("wrong GC content.")
        if tok in seen:
            fail("duplicate barcode.")
        seen.add(tok)
        masks.append(m)

    # pairwise distance (returns on first violation; feasible sets are tiny)
    L = len(masks)
    for i in range(L):
        mi = masks[i]
        for j in range(i + 1, L):
            if popcount(mi & masks[j]) > tmax:
                fail("distance violation.")

    F = L
    B = n // w
    if B < 1:
        B = 1

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
