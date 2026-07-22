# TIER: strong
# The insight: don't curve-fit a flexible shape to the quiet-day rows --
# IMPOSE the economic asymptote as a hypothesis-space constraint FIRST, then
# fit only the free constants it leaves open. Competition must drive shading
# to zero, so the shading term is forced into the saturating family
#     shade(n, rho) = v * (n-1) / ((n-1) + rho)          [-> v as n -> inf]
# and the appraisal-risk surcharge is forced to be a pure function of v alone
# (never of n):
#     risk(v, kappa) = kappa * (v - mu)**2 / mu
#
# For any FIXED rho, price - shade(n,rho) is linear in kappa with a known
# feature r = (v-mu)**2/mu, so kappa has a closed-form 1-D least-squares
# solution given rho. That collapses the whole fit to a 1-D scan over rho
# (coarse grid, then refined around the best point), each step an O(rows)
# closed-form kappa fit. Because the FUNCTIONAL FORM is right (not just the
# fit quality on the training range), the recovered (rho, kappa) generalise
# to n=25/60 and to appraisals the training ledger never showed.
import sys

MU = 100.0


def fit_kappa(train, rho):
    num = 0.0
    den = 0.0
    for n, v, p in train:
        shade = v * (n - 1) / ((n - 1) + rho)
        r = (v - MU) ** 2 / MU
        d = p - shade
        num += d * r
        den += r * r
    kappa = num / den if den > 1e-9 else 0.0
    sse = 0.0
    for n, v, p in train:
        shade = v * (n - 1) / ((n - 1) + rho)
        r = (v - MU) ** 2 / MU
        resid = (p - shade) - kappa * r
        sse += resid * resid
    return sse, kappa


def scan(train, lo, hi, steps):
    best = None
    for i in range(steps):
        rho = lo + (hi - lo) * i / (steps - 1)
        if rho <= 1e-6:
            continue
        sse, kappa = fit_kappa(train, rho)
        if best is None or sse < best[0]:
            best = (sse, rho, kappa)
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        print("v")
        return
    num_rows = int(data[0])
    vals = data[3:]
    train = []
    for i in range(num_rows):
        n = int(float(vals[3 * i]))
        v = float(vals[3 * i + 1])
        p = float(vals[3 * i + 2])
        train.append((n, v, p))

    if not train:
        print("v")
        return

    best = scan(train, 0.02, 8.0, 400)
    _, rho0, _ = best
    span = max(0.02, (8.0 - 0.02) / 400.0 * 2)
    lo2 = max(0.005, rho0 - span)
    hi2 = rho0 + span
    best2 = scan(train, lo2, hi2, 400)
    if best2 is not None and best2[0] < best[0]:
        best = best2

    _, rho, kappa = best
    print("v*(n-1)/((n-1)+%.10g) + %.10g*(v-%.10g)**2/%.10g" % (rho, kappa, MU, MU))


if __name__ == "__main__":
    main()
