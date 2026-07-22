import sys, math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def diffuse(u, steps, alpha):
    n = len(u)
    cur = [row[:] for row in u]
    for _ in range(steps):
        nxt = [[0.0] * n for _ in range(n)]
        for i in range(n):
            up = cur[(i - 1) % n]; dn = cur[(i + 1) % n]; me = cur[i]
            for j in range(n):
                lap = up[j] + dn[j] + me[(j - 1) % n] + me[(j + 1) % n] - 4.0 * me[j]
                nxt[i][j] = me[j] + alpha * lap
        cur = nxt
    return cur

def main():
    # ---- read instance ----
    try:
        toks = open(sys.argv[1]).read().split()
        it = iter(toks)
        N = int(next(it)); T = int(next(it))
        alpha = float(next(it)); cost = float(next(it))
        y = [[float(next(it)) for _ in range(N)] for _ in range(N)]
    except Exception:
        fail("bad input")

    # ---- baseline B = J(0) = ||y||^2 ----
    B = sum(v * v for row in y for v in row)
    B = max(1e-9, B)

    # ---- parse participant source field s (exactly N*N finite non-negative reals) ----
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if len(raw) != N * N:
        fail("expected %d values, got %d" % (N * N, len(raw)))
    s = [[0.0] * N for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(N):
            try:
                v = float(raw[k])
            except Exception:
                fail("non-numeric value")
            if not math.isfinite(v):
                fail("non-finite value")
            if v < 0.0:
                fail("negative source")
            s[i][j] = v
            k += 1

    # ---- objective J(s) = ||F(s) - y||^2 + cost * nz ----
    Fs = diffuse(s, T, alpha)
    fit = 0.0
    for i in range(N):
        for j in range(N):
            d = Fs[i][j] - y[i][j]
            fit += d * d
    nz = sum(1 for row in s for v in row if v > 1e-6)
    J = fit + cost * nz
    J = max(1e-12, J)

    sc = min(1000.0, 100.0 * B / J)
    print("fit=%.6f nz=%d J=%.6f B=%.6f Ratio: %.6f" % (fit, nz, J, B, sc / 1000.0))

if __name__ == "__main__":
    main()
