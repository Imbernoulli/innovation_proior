import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_instance(path):
    tok = open(path).read().split()
    it = iter(tok)
    N = int(next(it)); S = int(next(it)); T = int(next(it))
    L = int(next(it)); k = int(next(it))
    ch = int(next(it)); cv = int(next(it)); cr = int(next(it))
    lnum = int(next(it)); lden = int(next(it))
    return N, S, T, L, k, ch, cv, cr, lnum, lden

def longest_run_ok(seq, L):
    # True iff no maximal run of equal values exceeds L
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            run += 1
            if run > L:
                return False
        else:
            run = 1
    return True

def build_P(threading, treadling, tieup, N):
    # P[i][j] = tieup[treadling[i]-1][threading[j]-1]
    P = [None] * N
    for i in range(N):
        row = tieup[treadling[i]-1]
        P[i] = [row[threading[j]-1] for j in range(N)]
    return P

def floats_ok(P, N, L):
    for i in range(N):
        if not longest_run_ok(P[i], L):
            return False
    for j in range(N):
        col = [P[i][j] for i in range(N)]
        if not longest_run_ok(col, L):
            return False
    return True

def symmetry(P, N, ch, cv, cr):
    ah = av = ar = 0
    tot = N * N
    for i in range(N):
        Pi = P[i]
        Pmi = P[N-1-i]
        for j in range(N):
            v = Pi[j]
            if v == Pi[N-1-j]:
                ah += 1
            if v == Pmi[j]:
                av += 1
            if v == Pmi[N-1-j]:
                ar += 1
    fa = ah / tot; fv = av / tot; fr = ar / tot
    w = ch + cv + cr
    if w <= 0:
        w = 1
    return (ch*fa + cv*fv + cr*fr) / w

def diversity(P, N, k, S, T):
    # Distinct k x k windows, normalized by the structural ceiling. A window equals
    # tieup[treadling-window][threading-window], so at most S^k * T^k distinct windows
    # can EVER occur regardless of N -- that ceiling (not the number of window positions)
    # is the real budget, so the score is N-invariant and NOT trivially saturable.
    if k > N:
        return 0.0
    seen = set()
    W = N - k + 1
    for r in range(W):
        rows = P[r:r+k]
        for c in range(W):
            code = 0
            bit = 0
            for a in range(k):
                ra = rows[a]
                for b in range(c, c+k):
                    code |= (ra[b] & 1) << bit
                    bit += 1
            seen.add(code)
    ceil_windows = min(W * W, (S ** k) * (T ** k))
    return len(seen) / ceil_windows

def objective(P, N, k, S, T, ch, cv, cr, lnum, lden):
    sym = symmetry(P, N, ch, cv, cr)
    div = diversity(P, N, k, S, T)
    lam = lnum / lden
    return sym * (lam + (1.0 - lam) * div), sym, div

def baseline(N, S, T, k, ch, cv, cr, lnum, lden):
    # plain weave (tabby): threading/treadling straight on 2 shafts/treadles,
    # tie-up[t][s] = (s+t) mod 2. Uses only shafts/treadles 1,2 (S,T >= 2).
    threading = [1 + (j % 2) for j in range(N)]
    treadling = [1 + (i % 2) for i in range(N)]
    tieup = [[(s + t) % 2 for s in range(S)] for t in range(T)]
    P = build_P(threading, treadling, tieup, N)
    F, _, _ = objective(P, N, k, S, T, ch, cv, cr, lnum, lden)
    return F

def main():
    try:
        N, S, T, L, k, ch, cv, cr, lnum, lden = read_instance(sys.argv[1])
    except Exception:
        fail("bad input")

    try:
        tok = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    need = 2 * N + T * S
    if len(tok) != need:
        fail("token count %d != %d" % (len(tok), need))

    try:
        vals = [int(x) for x in tok]
    except Exception:
        fail("non-integer token")

    threading = vals[:N]
    treadling = vals[N:2*N]
    flat = vals[2*N:]

    for x in threading:
        if x < 1 or x > S:
            fail("threading out of range")
    for x in treadling:
        if x < 1 or x > T:
            fail("treadling out of range")
    tieup = []
    idx = 0
    for t in range(T):
        row = []
        for s in range(S):
            b = flat[idx]; idx += 1
            if b != 0 and b != 1:
                fail("tie-up bit not 0/1")
            row.append(b)
        tieup.append(row)

    P = build_P(threading, treadling, tieup, N)

    if not floats_ok(P, N, L):
        fail("float run exceeds L")

    F, sym, div = objective(P, N, k, S, T, ch, cv, cr, lnum, lden)
    B = baseline(N, S, T, k, ch, cv, cr, lnum, lden)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f sym=%.4f div=%.4f B=%.6f Ratio: %.6f" % (F, sym, div, B, sc / 1000.0))

if __name__ == "__main__":
    main()
