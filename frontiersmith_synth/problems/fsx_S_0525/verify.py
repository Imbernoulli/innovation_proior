#!/usr/bin/env python3
# Deterministic checker for "Thin-Film Coating: Hit a Reflectance Spectrum" (format C, minimize).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1]. Any feasibility violation -> Ratio: 0.0.
import sys, math

N0_FIXED = 1.0


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def reflectance(layers, ns, lam):
    # layers: list of (n, d) from incident side toward substrate. Normal-incidence TMM.
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
    r = (N0_FIXED - Y) / (N0_FIXED + Y)
    R = r.real * r.real + r.imag * r.imag
    return min(1.0, max(0.0, R))


def main():
    # ---- parse instance ----
    try:
        it = open(sys.argv[1]).read().split()
        p = 0
        n0 = float(it[p]); ns = float(it[p + 1]); p += 2
        M = int(it[p]); p += 1
        mats = [float(it[p + j]) for j in range(M)]; p += M
        K = int(it[p]); p += 1
        lams = []; Rstar = []
        for _ in range(K):
            lams.append(float(it[p])); Rstar.append(float(it[p + 1])); p += 2
        L = int(it[p]); lam0 = float(it[p + 1]); cost = float(it[p + 2]); dmax = float(it[p + 3])
    except Exception:
        fail("bad instance")

    # ---- parse participant output ----
    try:
        ot = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not ot:
        fail("empty output")

    try:
        m = int(ot[0])
    except Exception:
        fail("bad layer count")
    if m < 0 or m > L:
        fail("layer count out of range")
    if len(ot) < 1 + 2 * m:
        fail("truncated layers")

    layers = []
    for k in range(m):
        try:
            mi = int(ot[1 + 2 * k])
            d = float(ot[2 + 2 * k])
        except Exception:
            fail("bad layer %d" % k)
        if not math.isfinite(d):
            fail("non-finite thickness %d" % k)
        if mi < 0 or mi >= M:
            fail("material index out of range %d" % k)
        if d < 0.0 or d > dmax:
            fail("thickness out of range %d" % k)
        layers.append((mats[mi], d))

    # ---- objective (minimize) ----
    sse = 0.0
    for lam, rs in zip(lams, Rstar):
        R = reflectance(layers, ns, lam)
        e = R - rs
        sse += e * e
    F = sse + cost * m
    if not math.isfinite(F):
        fail("non-finite objective")

    # ---- baseline: bare substrate (zero layers), the checker's own trivial construction ----
    Rbare = ((N0_FIXED - ns) / (N0_FIXED + ns)) ** 2
    B = sum((Rbare - rs) ** 2 for rs in Rstar)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("SSE=%.6f layers=%d F=%.6f B=%.6f Ratio: %.6f" % (sse, m, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
