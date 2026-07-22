# TIER: strong
"""The insight: the observation process IS part of the model. A chunk of the
training rows are not really "hold time = T" -- they are "hold time >= T,
caller hung up". Throwing those rows away (or trusting their literal value)
is exactly the trap. Instead:

  1. Treat each row as a right-censored exponential draw with mean
     m(rho) = c / (1 - rho/rho_max): unknown SHAPE parameter rho_max (the
     pole location) and a scale c.
  2. For any CANDIDATE rho_max, c has a closed-form maximum-likelihood value
     (a "total time on test" statistic: sum of observed/censoring times over
     the regressor x=1/(1-rho/rho_max), divided by the number of genuinely
     OBSERVED -- i.e. uncensored -- events). This profiles out c exactly.
  3. That leaves a single 1-D search over rho_max: grid-search (coarse to
     fine) the profile negative log-likelihood, which folds in BOTH the
     uncensored hold times AND the censoring indicators (the fraction of
     capped calls at each load is exactly the extra signal that localises
     the singularity -- an ordinary regression on the observed values alone
     never sees it, because it never distinguishes "hung up at T" from
     "genuinely waited T").

The winning (rho_max, c) pair is emitted directly as the closed-form
divergence expression."""
import sys, math


def nll_given_rho_max(rho_max, rows, T):
    xs = []
    for rho, hold in rows:
        denom = 1.0 - rho / rho_max
        if denom <= 1e-6:
            return None, None
        xs.append(1.0 / denom)
    n_unc = 0
    total = 0.0
    for (rho, hold), x in zip(rows, xs):
        if hold >= T - 1e-6:
            total += T / x
        else:
            total += hold / x
            n_unc += 1
    if n_unc == 0:
        return None, None
    c = total / n_unc
    if c <= 1e-9:
        return None, None
    nll = 0.0
    for (rho, hold), x in zip(rows, xs):
        m = c * x
        if hold >= T - 1e-6:
            nll += T / m
        else:
            nll += math.log(m) + hold / m
    return nll, c


def fit(rows, T):
    max_train_rho = max(r for r, h in rows)
    lo, hi = max_train_rho * 1.001, max_train_rho * 8.0
    best = None
    for _round in range(6):
        n_grid = 60
        for i in range(n_grid + 1):
            frac = i / n_grid
            cand = lo + frac * (hi - lo)
            nll, c = nll_given_rho_max(cand, rows, T)
            if nll is None:
                continue
            if best is None or nll < best[0]:
                best = (nll, cand, c)
        if best is None:
            break
        span = (hi - lo) * 0.15
        lo = max(max_train_rho * 1.0005, best[1] - span)
        hi = best[1] + span
    if best is None:
        return 1.0, max_train_rho * 1.3
    return best[2], best[1]


def main():
    data = sys.stdin.read().split()
    n_train = int(data[0])
    T = float(data[2])
    idx = 3
    rows = []
    for _ in range(n_train):
        rho = float(data[idx]); hold = float(data[idx + 1])
        idx += 2
        rows.append((rho, hold))

    c_est, rho_max_est = fit(rows, T)

    expr = "( %.8f ) / abs ( 1 - rho / ( %.8f ) )" % (c_est, rho_max_est)
    print(expr)


if __name__ == "__main__":
    main()
