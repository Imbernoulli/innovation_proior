# TIER: strong
# Insight: node lines DIFFER across modes, so fuse ALL measured modes
# jointly instead of trusting any single one.
#
# For a candidate crack location x, the forward model says the fractional
# frequency shift of mode m is r_m ~= s * cos^2(m*pi*x/L). That is LINEAR in
# s for fixed x, so for every candidate x on a grid we solve the 1-D
# least-squares problem for the best-fit severity s(x) using every measured
# mode at once:
#     s(x) = sum_i r_i * c_i(x) / sum_i c_i(x)^2 ,   c_i(x) = cos^2(m_i*pi*x/L)
# A mode that is (near) its own node at this x has c_i(x) ~= 0 and
# automatically contributes ~nothing to the fit -- no special-casing needed,
# the least-squares weighting IS the "combine blind spots that don't
# overlap" insight. We take the residual-minimizing x as the primary
# location estimate.
#
# Because cos^2 is periodic/symmetric, several candidate x can fit the
# frequency data almost equally well (aliasing). We break that residual-tie
# ambiguity with the SECOND, complementary channel: the coarse damaged mode
# SHAPE data (frequency-shift-vs-shape). Among the near-optimal frequency
# candidates we pick the one whose predicted local shape dent best matches
# what was actually measured across every mode and gauge point.
import sys, math


def main():
    data = sys.stdin.read().split()
    p = 0
    t = int(data[p]); p += 1
    L = int(data[p]); p += 1
    G = int(data[p]); p += 1
    K = int(data[p]); p += 1
    modes = [int(data[p + i]) for i in range(K)]; p += K
    f0 = [float(data[p + i]) for i in range(K)]; p += K
    fdam = [float(data[p + i]) for i in range(K)]; p += K
    gpts = [float(data[p + i]) for i in range(G)]; p += G
    shape_u = []
    for i in range(K):
        shape_u.append([float(data[p + j]) for j in range(G)]); p += G
    shape_d = []
    for i in range(K):
        shape_d.append([float(data[p + j]) for j in range(G)]); p += G

    S_MAX_OUT = 0.5
    r = [1.0 - fdam[i] / f0[i] for i in range(K)]

    NGRID = 1600
    cand_list = []
    for gi in range(NGRID + 1):
        x = L * gi / NGRID
        cs = [math.cos(m * math.pi * x / L) ** 2 for m in modes]
        num = sum(r[i] * cs[i] for i in range(K))
        den = sum(cs[i] * cs[i] for i in range(K))
        s_x = 0.0 if den < 1e-9 else num / den
        s_x = max(0.0, min(S_MAX_OUT, s_x))
        resid = sum((r[i] - s_x * cs[i]) ** 2 for i in range(K))
        cand_list.append((resid, x, s_x))

    cand_list.sort(key=lambda z: z[0])
    top_resid = cand_list[0][0]
    tol = top_resid * 0.01 + 1e-4
    near = [c for c in cand_list if c[0] <= top_resid + tol]
    near.sort(key=lambda z: z[1])

    # collapse near-duplicate x's (keep the best residual within each cluster)
    picked = []
    for c in near:
        if not picked or abs(c[1] - picked[-1][1]) > L * 0.02:
            picked.append(c)
        elif c[0] < picked[-1][0]:
            picked[-1] = c
    if len(picked) > 8:
        picked.sort(key=lambda z: z[0])
        picked = picked[:8]

    w = 0.10 * L

    def shape_cost(x, s):
        cost = 0.0
        n = 0
        for i in range(K):
            for g in range(G):
                xg = gpts[g]
                bump = math.exp(-((xg - x) / w) ** 2)
                pred = shape_u[i][g] * (1.0 - s * bump)
                cost += (shape_d[i][g] - pred) ** 2
                n += 1
        return cost / max(1, n)

    best = None
    for (resid, x, s_x) in picked:
        sc = shape_cost(x, s_x)
        if best is None or sc < best[0]:
            best = (sc, x, s_x)

    x_hat, s_hat = best[1], best[2]
    print("%.6f %.6f" % (x_hat, s_hat))


if __name__ == "__main__":
    main()
