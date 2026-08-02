# TIER: strong
# The insight: raw observed tenure is NOT a fair estimate of anything once
# customers are right-censored -- but the HAZARD SHAPE is still recoverable.
# For each cohort bucket, build the Nelson-Aalen cumulative-hazard estimate
# H(t) = sum d_k/n_k over event times, where n_k is the RISK SET (everyone
# still observed, censored or not, up to their own exit/censor time) -- this
# correctly discounts each event by how many customers could still have
# exited at that moment, which is exactly what raw averaging fails to do.
# If the hidden law is Weibull-shaped, log H(t) is LINEAR in log(t) with
# slope = kappa and intercept = -kappa*log(lambda); that line is estimable
# entirely from data inside the visible window [0, T_obs], and -- unlike the
# mean tenure -- it can be safely EXTRAPOLATED past T_obs, because a shape
# estimated from where hazard has already been observed doesn't need to see
# the tail to describe it. Cohorts too sparse for their own fit borrow the
# pooled (x-ignoring) fit; the per-bucket (kappa, log lambda) estimates are
# then regressed against the cohort covariate x to recover cohort
# heterogeneity, so the model generalises to cohorts sparsely (or never)
# observed at train time too.
import sys, math, bisect

BUCKETS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def nelson_aalen(pairs):
    """pairs: list of (observed_tenure, censored). Returns [(t_k, H(t_k))..] at
    each distinct event time, respecting the risk set (censoring-corrected)."""
    if not pairs:
        return []
    sorted_obs = sorted(o for o, c in pairs)
    N = len(pairs)
    event_counts = {}
    for o, c in pairs:
        if c == 0:
            event_counts[o] = event_counts.get(o, 0) + 1
    events = sorted(event_counts.keys())
    H = 0.0
    out = []
    for tk in events:
        n_k = N - bisect.bisect_left(sorted_obs, tk)
        d_k = event_counts[tk]
        if n_k > 0:
            H += d_k / n_k
        out.append((tk, H))
    return out


def ols(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    sx = sum(xs); sy = sum(ys)
    sxx = sum(v * v for v in xs); sxy = sum(u * v for u, v in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def weibull_fit(pairs):
    """Nelson-Aalen -> log-log regression -> (kappa, lambda), or None."""
    na = nelson_aalen(pairs)
    pts = [(math.log(tk), math.log(H)) for tk, H in na if H > 1e-9 and tk > 0]
    fit = ols([p[0] for p in pts], [p[1] for p in pts])
    if fit is None:
        return None
    slope, intercept = fit
    if slope <= 1e-4:
        return None
    kappa = slope
    lam = math.exp(-intercept / kappa)
    if not (lam > 1e-6):
        return None
    return kappa, lam


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("0.5"); return
    N = int(data[0])
    vals = data[3:]
    rows = []
    for i in range(N):
        x = float(vals[3 * i]); obs = float(vals[3 * i + 1]); cens = int(vals[3 * i + 2])
        rows.append((x, obs, cens))

    pooled_fit = weibull_fit([(o, c) for _, o, c in rows]) or (1.0, max(1e-6, sum(o for _, o, c in rows) / max(1, N)))

    per_bucket = []
    for b in BUCKETS:
        pairs = [(o, c) for x, o, c in rows if abs(x - b) < 1e-6]
        fit = weibull_fit(pairs) if len(pairs) >= 4 else None
        kappa, lam = fit if fit is not None else pooled_fit
        per_bucket.append((b, kappa, lam))

    bxs = [b for b, k, l in per_bucket]
    kappas = [k for b, k, l in per_bucket]
    loglams = [math.log(l) for b, k, l in per_bucket]

    fit_k = ols(bxs, kappas)
    K0, K1 = (kappas[0], 0.0) if fit_k is None else (fit_k[1], fit_k[0])
    fit_l = ols(bxs, loglams)
    LOGL0, L1 = (loglams[0], 0.0) if fit_l is None else (fit_l[1], fit_l[0])
    L0 = math.exp(LOGL0)

    print("exp ( - ( ( t / ( %.6f * exp ( %.6f * x ) ) ) ** ( %.6f + %.6f * x ) ) )"
          % (L0, L1, K0, K1))


if __name__ == "__main__":
    main()
