import sys, random
import numpy as np

# ---------------------------------------------------------------------------
# laffer-ledge-tax generator
#
# Population of guild workers. Each worker i has:
#   m_i  : base taxable income (pre-tax income at a zero marginal rate)
#   e_i  : elasticity of taxable income (constant-elasticity best response)
#   f_i  : fixed participation cost (utility units)
#
# Constant-elasticity best response to a piecewise-linear schedule T(.):
#   inside a bracket with marginal rate tau the interior optimum earns
#     z*(tau) = m * (1 - tau) ** e
#   disutility of earning z is  v(z) = (m/(1+1/e)) * (z/m) ** (1+1/e)
#   utility if working is   U = (z - T(z)) - v(z) - f   (work iff U > 0)
#
# The elasticity distribution is planted BIMODAL and correlated with income:
# a mass of highly-elastic workers sits squarely in the MIDDLE income band,
# while the bottom and top income bands are inelastic.  The revenue-optimal
# marginal-rate schedule must therefore DIP in the middle (a non-monotone
# shape no progressive/flat/linear template produces).
# ---------------------------------------------------------------------------

RATE_MAX = 0.95


def evaluate(m, e, f, thresholds, rates):
    """Vectorized: return (revenue, welfare) for a schedule on the population."""
    K = len(rates)
    thr = np.asarray(thresholds, dtype=float)
    rt = np.asarray(rates, dtype=float)
    inv = 1.0 + 1.0 / e
    Tcum = np.zeros(K)
    for k in range(1, K):
        Tcum[k] = Tcum[k - 1] + rt[k - 1] * (thr[k] - thr[k - 1])
    edges = np.concatenate([thr, [np.inf]])

    def tax_of(z):
        idx = np.searchsorted(edges, z, side='right') - 1
        idx = np.clip(idx, 0, K - 1)
        return Tcum[idx] + rt[idx] * (z - thr[idx])

    def util(z):
        v = (m / inv) * np.power(z / m, inv)
        return (z - tax_of(z)) - v

    best_u = np.zeros_like(m)
    best_z = np.zeros_like(m)
    have = np.zeros(len(m), dtype=bool)

    def consider(z, valid):
        nonlocal best_u, best_z, have
        u = util(z)
        take = valid & (~have | (u > best_u + 1e-12))
        best_u = np.where(take, u, best_u)
        best_z = np.where(take, z, best_z)
        have = have | valid

    for k in range(K):
        zc = m * np.power(1.0 - rt[k], e)
        hi = edges[k + 1]
        valid = (zc >= thr[k]) & (zc <= hi) & (zc > 0)
        consider(zc, valid)
    for k in range(1, K):
        zc = np.full_like(m, thr[k])
        consider(zc, zc > 0)

    up = best_u - f
    works = have & (up > 1e-12)
    R = float(np.sum(np.where(works, tax_of(best_z), 0.0)))
    W = float(np.sum(np.where(works, up, 0.0)))
    return R, W


def main():
    tid = int(sys.argv[1])
    rng = random.Random(90000 + tid)

    N_list = [150, 300, 600, 1200, 2500, 5000, 10000, 20000, 35000, 50000]
    N = N_list[(tid - 1) % 10]

    # sharper elasticity contrast (harder trap) for higher testIds
    e_high = 0.85 + 0.09 * min(tid, 7)      # 0.94 .. 1.48
    e_low = 0.25
    p_low, p_mid, p_top = 0.40, 0.35, 0.25   # elastic middle mass = 35%

    rows = []
    for _ in range(N):
        r = rng.random()
        if r < p_low:
            m = rng.uniform(5.0, 25.0)
            e = e_low + rng.uniform(-0.03, 0.03)
        elif r < p_low + p_mid:
            m = rng.uniform(30.0, 55.0)          # MIDDLE income band
            e = e_high + rng.uniform(-0.05, 0.05)  # ELASTIC mass
        else:
            m = rng.uniform(60.0, 100.0)
            e = e_low + rng.uniform(-0.03, 0.03)
        e = max(0.12, e)
        f = rng.uniform(0.0, 2.5)
        rows.append((m, e, f))

    m = np.array([x[0] for x in rows])
    e = np.array([x[1] for x in rows])
    f = np.array([x[2] for x in rows])

    # welfare floor = phi * (welfare achieved by a mild flat reference rate)
    _, W_ref = evaluate(m, e, f, [0.0], [0.30])
    phi = 0.72
    floor = phi * W_ref

    out = ["%d %.6f" % (N, floor)]
    for (mm, ee, ff) in rows:
        out.append("%.6f %.6f %.6f" % (mm, ee, ff))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
