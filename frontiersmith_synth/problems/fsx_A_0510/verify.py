import sys, random

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_tokens(path, maxbytes):
    with open(path, "rb") as f:
        return f.read(maxbytes).split()

def main():
    # ---- parse instance (generator-controlled, small) ----
    tk = read_tokens(sys.argv[1], 16_000_000)
    it = iter(tk)
    try:
        t = int(next(it))
        ms = [int(next(it)) for _ in range(t)]
        k = int(next(it))
        A = []
        for i in range(t):
            s = int(next(it))
            ai = set(int(next(it)) % ms[i] for _ in range(s))
            A.append(ai)
    except StopIteration:
        fail("bad input")
    n = 1
    for m in ms:
        n *= m

    def inT(d):
        for m, a in zip(ms, A):
            if (d % m) not in a:
                return False
        return True

    def coverage(B):
        cov = set()
        L = len(B)
        for i in range(L):
            bi = B[i]
            for j in range(L):
                d = (bi - B[j]) % n
                if inT(d):
                    cov.add(d)
        return len(cov)

    # ---- internal baseline B: a fixed-seed random size-k residue set ----
    rng = random.Random(987654321)
    B0 = rng.sample(range(n), k)
    base = max(1, coverage(B0))

    # ---- parse participant output (adversarially; bound the read) ----
    ot = read_tokens(sys.argv[2], 8_000_000)
    if len(ot) < 1:
        fail("empty output")
    try:
        j = int(ot[0])
    except Exception:
        fail("bad count token")
    if j < 0 or j > k:
        fail("count out of range")
    if len(ot) < 1 + j:
        fail("not enough residues")
    B = []
    seen = set()
    for x in range(1, 1 + j):
        try:
            v = int(ot[x])           # rejects nan/inf/garbage (int() raises)
        except Exception:
            fail("bad residue token")
        if v < 0 or v >= n:
            fail("residue out of range")
        if v in seen:
            fail("duplicate residue")
        seen.add(v)
        B.append(v)

    F = coverage(B)
    sc = min(1000.0, 100.0 * F / max(1e-9, base))
    print("F=%d B=%d Ratio: %.6f" % (F, base, sc / 1000.0))

if __name__ == "__main__":
    main()
