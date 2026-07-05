import sys, random

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

# ---------- exact objective (integer arithmetic only) ----------
def gram(M, n):
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s += M[k][i] * M[k][j]
            G[i][j] = s
    return G

def energy_from_G(G, n):
    return sum(G[i][j] * G[i][j] for i in range(n) for j in range(i + 1, n))

def energy(M, n):
    return energy_from_G(gram(M, n), n)

# ---------- internal baseline B (must match solutions/trivial.py) ----------
def descend(M, G, n, rng, iters):
    """Deterministic single-flip greedy descent on rows 1..n-1 (row 0 is fixed)."""
    E = energy_from_G(G, n)
    for _ in range(iters):
        r = rng.randrange(1, n)
        c = rng.randrange(n)
        v = M[r][c]
        dE = 0
        for j in range(n):
            if j == c:
                continue
            new = G[c][j] - 2 * v * M[r][j]
            dE += new * new - G[c][j] * G[c][j]
        if dE <= 0:
            for j in range(n):
                if j == c:
                    continue
                new = G[c][j] - 2 * v * M[r][j]
                G[c][j] = new
                G[j][c] = new
            M[r][c] = -v
            E += dE
    return E

def baseline_energy(n, r0):
    rng = random.Random(90000 + n)
    M = [list(r0)] + [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n - 1)]
    G = gram(M, n)
    return descend(M, G, n, rng, 3 * n)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    # ---- parse instance ----
    try:
        it = iter(inp)
        n = int(next(it))
        r0 = [int(next(it)) for _ in range(n)]
    except Exception:
        fail("bad input")
    for v in r0:
        if v not in (-1, 1):
            fail("bad instance row")

    # ---- parse participant matrix: exactly n*n values, each +/-1 ----
    try:
        vals = [int(t) for t in out]
    except Exception:
        fail("non-integer output")
    if len(vals) != n * n:
        fail("expected %d entries, got %d" % (n * n, len(vals)))
    for v in vals:
        if v not in (-1, 1):
            fail("entry %d not in {-1,+1}" % v)
    M = [vals[i * n:(i + 1) * n] for i in range(n)]

    # ---- feasibility: fixed first row ----
    if M[0] != r0:
        fail("first row does not match the pre-provisioned beacon code")

    # ---- objective + baseline ----
    F = energy(M, n)
    B = baseline_energy(n, r0)
    sc = min(1000.0, 100.0 * B / max(1, F))
    print("E=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
