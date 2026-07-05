import sys
import math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def linearity_numpy(S, n):
    import numpy as np
    N = 1 << n
    Sarr = np.asarray(S, dtype=np.int64)
    pc = np.zeros(N, dtype=np.int64)
    for i in range(1, N):
        pc[i] = pc[i >> 1] + (i & 1)
    Lmax = 0
    for b in range(1, N):
        parity = pc[b & Sarr] & 1
        f = (1 - 2 * parity).astype(np.float64)  # (-1)^{b . S(x)}
        h = 1
        while h < N:
            f = f.reshape(-1, 2 * h)
            a = f[:, :h].copy()
            c = f[:, h:2 * h].copy()
            f[:, :h] = a + c
            f[:, h:2 * h] = a - c
            f = f.reshape(-1)
            h *= 2
        m = int(np.max(np.abs(f)))
        if m > Lmax:
            Lmax = m
    return Lmax


def linearity_pure(S, n):
    N = 1 << n
    pc = [0] * N
    for i in range(1, N):
        pc[i] = pc[i >> 1] + (i & 1)
    Lmax = 0
    for b in range(1, N):
        f = [1 - 2 * (pc[b & S[x]] & 1) for x in range(N)]
        h = 1
        while h < N:
            for i in range(0, N, 2 * h):
                for j in range(i, i + h):
                    a = f[j]
                    c = f[j + h]
                    f[j] = a + c
                    f[j + h] = a - c
            h *= 2
        m = max(abs(v) for v in f)
        if m > Lmax:
            Lmax = m
    return Lmax


def linearity(S, n):
    try:
        return linearity_numpy(S, n)
    except Exception:
        return linearity_pure(S, n)


def main():
    # ---- read instance ----
    try:
        inp = open(sys.argv[1]).read().split()
        n = int(inp[0])
    except Exception:
        fail("bad input")
    if n < 2 or n > 12:
        fail("n out of range")
    N = 1 << n

    # ---- internal baseline B: the identity permutation is affine, so every
    #      nonzero output bit is a linear function whose Walsh spectrum has a
    #      single peak of full height 2^n.  Thus L(identity) = 2^n = B. ----
    B = float(N)

    # ---- parse participant output: exactly N integers, S(0..N-1) ----
    raw = open(sys.argv[2]).read().split()
    if len(raw) != N:
        fail("expected %d integers, got %d" % (N, len(raw)))
    S = [0] * N
    seen = [False] * N
    for x in range(N):
        tok = raw[x]
        try:
            v = int(tok)
        except Exception:
            fail("non-integer token %r" % tok)
        # reject any non-finite disguised as float text
        fv = float(v)
        if not math.isfinite(fv):
            fail("non-finite value")
        if v < 0 or v >= N:
            fail("value %d out of range [0,%d)" % (v, N))
        if seen[v]:
            fail("not a permutation: repeated value %d" % v)
        seen[v] = True
        S[x] = v

    # ---- objective: minimize linearity L = max_{b!=0, a} |W_S(a,b)| ----
    L = linearity(S, n)
    if L <= 0:
        fail("degenerate spectrum")

    # minimization normalization (smaller linearity -> higher score)
    sc = min(1000.0, 100.0 * B / max(1e-9, float(L)))
    nl = (N >> 1) - L // 2  # reported nonlinearity for readability
    print("n=%d L=%d NL=%d B=%d Ratio: %.6f" % (n, L, nl, int(B), sc / 1000.0))


if __name__ == "__main__":
    main()
