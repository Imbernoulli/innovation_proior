# TIER: strong
# Insight: the reflectance objective is hopelessly nonlinear layer-by-layer, but in
# OPTICAL-ADMITTANCE space a quarter-wave layer of index n performs the impedance
# transformation Y -> n^2 / Y at the design wavelength. Composed, alternating high/low
# quarter-wave layers move admittance MULTIPLICATIVELY (additively in log-admittance) --
# that additive ladder is exactly what builds a reflectance BAND. So instead of chasing
# one wavelength at a time, we enumerate the admittance-motivated structural family
# (alternating high/low quarter-wave stacks over every material pair, every period count,
# both start orders, plus 1-2 layer anti-reflection combos), then refine each candidate's
# thicknesses by deterministic coordinate descent against the FULL band, and keep the
# design with the smallest penalized objective F = SSE + cost * layers.
import sys, math

N0 = 1.0


def reflectance(layers, ns, lam):
    m00, m01, m10, m11 = 1 + 0j, 0j, 0j, 1 + 0j
    for (n, d) in layers:
        delta = 2.0 * math.pi * n * d / lam
        c = math.cos(delta); s = math.sin(delta)
        a00, a01, a10, a11 = c, 1j * s / n, 1j * n * s, c
        m00, m01, m10, m11 = (m00 * a00 + m01 * a10, m00 * a01 + m01 * a11,
                              m10 * a00 + m11 * a10, m10 * a01 + m11 * a11)
    B = m00 * 1.0 + m01 * ns
    C = m10 * 1.0 + m11 * ns
    Y = C / B
    r = (N0 - Y) / (N0 + Y)
    R = r.real * r.real + r.imag * r.imag
    return min(1.0, max(0.0, R))


def sse(ds, nlist, ns, lams, Rstar):
    s = 0.0
    layers = list(zip(nlist, ds))
    for lam, rs in zip(lams, Rstar):
        e = reflectance(layers, ns, lam) - rs
        s += e * e
    return s


def refine(nlist, ds, ns, lams, Rstar, dmax):
    # deterministic coordinate descent on thicknesses over a shrinking multiplicative grid
    ds = list(ds)
    cur = sse(ds, nlist, ns, lams, Rstar)
    steps = [0.5, 0.25, 0.12, 0.06, 0.03]
    for step in steps:
        improved = True
        sweeps = 0
        while improved and sweeps < 3:
            improved = False
            sweeps += 1
            for i in range(len(ds)):
                base = ds[i]
                for cand in (base * (1.0 + step), base * (1.0 - step),
                             base + step * 40.0, base - step * 40.0):
                    if cand <= 0.0 or cand > dmax:
                        continue
                    old = ds[i]; ds[i] = cand
                    s = sse(ds, nlist, ns, lams, Rstar)
                    if s + 1e-15 < cur:
                        cur = s; improved = True
                    else:
                        ds[i] = old
    return ds, cur


def main():
    it = sys.stdin.read().split()
    p = 0
    n0 = float(it[p]); ns = float(it[p + 1]); p += 2
    M = int(it[p]); p += 1
    mats = [float(it[p + j]) for j in range(M)]; p += M
    K = int(it[p]); p += 1
    lams = []; Rstar = []
    for _ in range(K):
        lams.append(float(it[p])); Rstar.append(float(it[p + 1])); p += 2
    L = int(it[p]); lam0 = float(it[p + 1]); cost = float(it[p + 2]); dmax = float(it[p + 3])

    def qw(n):
        return lam0 / (4.0 * n)

    # ---- build the candidate structural family ----
    candidates = []  # each: list of material-index sequence
    idx = list(range(M))
    # alternating high/low quarter-wave stacks over every ORDERED material pair
    for a in idx:
        for b in idx:
            if a == b:
                continue
            seq = []
            for t in range(L):
                seq.append(a if (t % 2 == 0) else b)
                if len(seq) == L:
                    break
                # allow any truncated length; we also try shorter versions below
            # try every prefix length 1..L of this alternation
            for ln in range(1, L + 1):
                candidates.append(seq[:ln])
    # single/double-layer AR combos (already covered by short alternations, but add pure singles)
    for a in idx:
        candidates.append([a])

    # dedup
    seen = set(); uniq = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key); uniq.append(c)
    candidates = uniq

    best = None  # (F, seq, ds)
    for seq in candidates:
        nlist = [mats[i] for i in seq]
        ds0 = [qw(n) for n in nlist]
        ds, s = refine(nlist, ds0, ns, lams, Rstar, dmax)
        F = s + cost * len(seq)
        if best is None or F < best[0]:
            best = (F, seq, ds)

    F, seq, ds = best
    out = [str(len(seq))]
    for i, d in zip(seq, ds):
        out.append("%d %.6f" % (i, d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
