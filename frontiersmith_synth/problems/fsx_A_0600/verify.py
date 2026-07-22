import sys, math

# fsx_A_0600 -- epidemic-threshold-eigendrop  (Format C, minimize spectral radius)
# CLI: python3 verify.py <in> <out> <ans>   (ans ignored)
# Score is deterministic: power iteration with fixed start + fixed iteration count.

ITERS = 500


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def spectral_radius(n, elist):
    # Largest eigenvalue of the symmetric nonnegative adjacency matrix, via
    # deterministic power iteration from the all-ones vector (bit-for-bit stable).
    if n == 0:
        return 0.0
    x = [1.0] * n
    s = math.sqrt(float(n))
    x = [v / s for v in x]
    for _ in range(ITERS):
        y = [0.0] * n
        for (i, j, w) in elist:
            y[i] += w * x[j]
            y[j] += w * x[i]
        nrm = 0.0
        for v in y:
            nrm += v * v
        nrm = math.sqrt(nrm)
        if nrm <= 0.0:
            return 0.0
        inv = 1.0 / nrm
        x = [v * inv for v in y]
    # Rayleigh quotient lambda = x^T A x  (x already unit-norm)
    y = [0.0] * n
    for (i, j, w) in elist:
        y[i] += w * x[j]
        y[j] += w * x[i]
    lam = 0.0
    for t in range(n):
        lam += x[t] * y[t]
    return lam


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("no input")
    try:
        it = iter(inp)
        n = int(next(it)); m = int(next(it)); k = int(next(it))
        edges = []
        for _ in range(m):
            u = int(next(it)) - 1
            v = int(next(it)) - 1
            w = int(next(it))
            edges.append((u, v, w))
    except Exception:
        fail("bad input")

    # ---- parse participant output: distinct edge indices in 1..m, count <= k ----
    try:
        toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    chosen = []
    seen = set()
    for tk in toks:
        try:
            idx = int(tk)          # rejects nan/inf/garbage tokens
        except Exception:
            fail("non-integer token %r" % tk)
        if idx < 1 or idx > m:
            fail("index out of range %d" % idx)
        if idx in seen:
            fail("duplicate index %d" % idx)
        seen.add(idx)
        chosen.append(idx)
    if len(chosen) > k:
        fail("over budget %d>%d" % (len(chosen), k))

    # ---- objective: spectral radius after removing the chosen edges ----
    orig_lam = spectral_radius(n, edges)
    if orig_lam <= 1e-9:
        fail("degenerate instance")

    rm = set(i - 1 for i in chosen)   # 0-based positions into edges
    kept = [edges[i] for i in range(m) if i not in rm]
    part_lam = spectral_radius(n, kept)
    drop_part = orig_lam - part_lam
    if drop_part < 0.0:
        drop_part = 0.0

    # ---- internal baseline B: remove the FIRST k edges (indices 1..k) ----
    base_rm = set(range(min(k, m)))
    base_kept = [edges[i] for i in range(m) if i not in base_rm]
    base_lam = spectral_radius(n, base_kept)
    drop_base = orig_lam - base_lam
    if drop_base < 1e-9:
        drop_base = 1e-9

    # minimize spectral radius == maximize eigen-drop; normalize by baseline drop.
    sc = min(1000.0, 100.0 * drop_part / drop_base)
    print("orig=%.6f part=%.6f base=%.6f drop_part=%.6f drop_base=%.6f"
          % (orig_lam, part_lam, base_lam, drop_part, drop_base))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
