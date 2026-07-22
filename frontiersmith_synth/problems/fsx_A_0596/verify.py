import sys

# Deterministic scorer for the datum-tree error-budget problem (format C, minimize).
#
# Instance: n features (0 = master datum / root), per-feature base op error a[i]
# and precise op error p[i] (p<a), a list of allowed datums for each feature
# (indices < i), a precise-slot budget k, and C weighted critical pairs.
#
# Submission (stdout):
#   line 1: par[0..n-1]   datum of each feature; par[0] == -1
#   line 2: q  f_1 .. f_q  the (<=k) features whose op uses the precise machine
#
# The datum map is a rooted forest; each non-root feature i contributes one edge
# i->par[i] whose "op error" is p[i] if i is precise else a[i]. For a critical
# pair (u,v) the worst-case relative error is the sum of op errors along the
# unique tree path between u and v (all nodes on the path except their LCA). The
# objective F = max over pairs of w * patherror. Lower is better.

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_instance(path):
    with open(path) as f:
        tok = f.read().split()
    it = iter(tok)
    n = int(next(it)); k = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    p = [int(next(it)) for _ in range(n)]
    allowed = []
    for i in range(n):
        d = int(next(it))
        allowed.append([int(next(it)) for _ in range(d)])
    C = int(next(it))
    pairs = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(C)]
    return n, k, a, p, allowed, pairs

def objective(n, par, eff, pairs):
    # depth via par (par[i] < i guaranteed for feasible trees)
    depth = [0] * n
    for i in range(1, n):
        depth[i] = depth[par[i]] + 1
    best = 0
    for (u, v, w) in pairs:
        x, y = u, v; s = 0
        while depth[x] > depth[y]:
            s += eff[x]; x = par[x]
        while depth[y] > depth[x]:
            s += eff[y]; y = par[y]
        while x != y:
            s += eff[x] + eff[y]; x = par[x]; y = par[y]
        c = w * s
        if c > best:
            best = c
    return best

def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n, k, a, p, allowed, pairs = read_instance(inf)

    # ---- parse submission strictly ----
    try:
        with open(outf) as f:
            raw = f.read().split()
    except Exception:
        fail("no output")
    if len(raw) < n + 1:
        fail("too few tokens")
    try:
        vals = [int(t) for t in raw]      # rejects nan/inf/floats/garbage
    except Exception:
        fail("non-integer token")

    par = vals[:n]
    q = vals[n]
    if q < 0 or q > k:
        fail("bad precise count")
    need = n + 1 + q
    if len(vals) != need:
        fail("token count mismatch (extra/short)")
    precise = vals[n + 1: n + 1 + q]

    # ---- feasibility of the datum tree ----
    if par[0] != -1:
        fail("root must have datum -1")
    aset = [set(al) for al in allowed]
    for i in range(1, n):
        if par[i] == -1 or par[i] not in aset[i]:
            fail("feature %d has an illegal datum" % i)
    # par[i] in allowed[i] with allowed indices < i => automatically acyclic tree

    pset = set()
    for f in precise:
        if f < 1 or f >= n:
            fail("precise index out of range")
        if f in pset:
            fail("duplicate precise slot")
        pset.add(f)

    # ---- objective ----
    eff = [p[i] if i in pset else a[i] for i in range(n)]
    F = objective(n, par, eff, pairs)
    if F <= 0:
        fail("degenerate objective")

    # ---- internal baseline: first-listed datum, no precise slots ----
    par_b = [-1] + [allowed[i][0] for i in range(1, n)]
    eff_b = a[:]
    B = objective(n, par_b, eff_b, pairs)
    if B <= 0:
        B = 1

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
