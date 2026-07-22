# TIER: greedy
# The obvious approach: repeatedly find the wavelength where achieved reflectance is
# farthest from the target and drop in a quarter-wave layer (thickness lambda_worst/(4n))
# of whichever material most reduces the error AT THAT wavelength -- treating each
# wavelength independently. Accept the layer only if it lowers total SSE, else stop.
#
# This never sees that every layer couples ALL wavelengths at once, so it cannot build the
# periodic admittance structure that produces a reflectance band. It plateaus early.
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


def sse(layers, ns, lams, Rstar):
    s = 0.0
    for lam, rs in zip(lams, Rstar):
        e = reflectance(layers, ns, lam) - rs
        s += e * e
    return s


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

    layers = []
    cur = sse(layers, ns, lams, Rstar)
    for _ in range(L):
        # wavelength with the largest current error
        wk = max(range(K), key=lambda k: abs(reflectance(layers, ns, lams[k]) - Rstar[k]))
        lw = lams[wk]
        best = None
        for mi in range(M):
            n = mats[mi]
            d = lw / (4.0 * n)  # quarter-wave AT the worst wavelength
            if d > dmax:
                continue
            trial = layers + [(n, d)]
            s = sse(trial, ns, lams, Rstar)
            if best is None or s < best[0]:
                best = (s, mi, d)
        if best is None:
            break
        s, mi, d = best
        if s + 1e-12 >= cur:  # no improvement in total error -> stop
            break
        layers.append((mats[mi], d))
        cur = s

    out = [str(len(layers))]
    # recover the material index for each stored (n,d)
    for (n, d) in layers:
        mi = min(range(M), key=lambda j: abs(mats[j] - n))
        out.append("%d %.6f" % (mi, d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
