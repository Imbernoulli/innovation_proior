# TIER: strong
# ESTIMATION BEFORE CONTROL.  For each of the six lines, run a cheap online
# hypothesis test on its 30-week history to classify which demand family it is
# living in (intermittent / bursty / seasonal / trending / flat-or-unclassified),
# THEN pick the order-up-to recipe suited to that family:
#
#   - The buffer size in every branch is the classical single-period newsvendor
#     critical-fractile quantile: crit = stockout_cost / (stockout_cost +
#     holding_cost) read straight from the instance, and the buffer is the
#     empirical crit-quantile of the relevant demand window (NOT an arbitrary
#     fixed "mean + 1 std"), so a line whose true cost tradeoff favors heavy
#     safety stock (stockout >> holding) automatically gets one, and a cheap
#     line does not carry it needlessly.
#   - seasonal:  keep the 12-phase SHAPE (phase means) and lift the whole shape by
#     the newsvendor buffer computed from the pooled history.
#   - trending:  detrend via a simple linear fit, extrapolate the fitted line
#     (trend=slope) and add the newsvendor buffer from the *residual* noise.
#   - bursty / intermittent / everything else: the newsvendor quantile of the raw
#     30-week history already captures "some weeks are far above the mean" far
#     better than a Gaussian mean+z*std guess, because it is taken directly from
#     the observed (possibly heavy-tailed / zero-inflated) empirical distribution.
#
# On top of the classification a SMALL always-on reactive term (react) nudges next
# week's target upward whenever last week's realized demand ran above the line's
# historical mean. History alone cannot reveal a regime change that only starts
# after the observation window ends -- so instead of freezing forever on the
# calibration-window classification, the policy keeps a lightweight closed-loop
# correction running through the whole future horizon.  This is what lets it
# partially recover on a line that looked calm during history and then shifts.
import sys, json, math

inst = json.load(sys.stdin)
period = inst["period"]


def stats(hist):
    n = len(hist)
    mean = sum(hist) / float(n)
    var = sum((x - mean) ** 2 for x in hist) / float(n)
    zero_frac = sum(1 for x in hist if x == 0) / float(n)
    tbar = (n - 1) / 2.0
    num = sum((t - tbar) * (hist[t] - mean) for t in range(n))
    den = sum((t - tbar) ** 2 for t in range(n))
    slope = num / den if den > 1e-9 else 0.0
    intercept = mean - slope * tbar
    ssres = sum((hist[t] - (intercept + slope * t)) ** 2 for t in range(n))
    sstot = n * var
    r2 = 1.0 - ssres / sstot if sstot > 1e-9 else 0.0
    groups = {}
    for t, x in enumerate(hist):
        groups.setdefault(t % period, []).append(x)
    phase_mean = {p: sum(v) / len(v) for p, v in groups.items()}
    between = sum((phase_mean[p] - mean) ** 2 * len(groups[p]) for p in groups) / n
    seasonal_strength = between / var if var > 1e-9 else 0.0
    cv2 = var / (mean * mean) if mean > 1e-9 else 999.0
    return dict(mean=mean, var=var, zero_frac=zero_frac, slope=slope,
                intercept=intercept, r2=r2, phase_mean=phase_mean,
                seasonal_strength=seasonal_strength, cv2=cv2)


def quantile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return float(s[idx])


policies = []
for tr in inst["traces"]:
    hist = tr["history"]
    n = len(hist)
    st = stats(hist)
    stockout_cost = tr["stockout_cost"]; holding_cost = tr["holding_cost"]
    crit = stockout_cost / (stockout_cost + holding_cost) if (stockout_cost + holding_cost) > 1e-9 else 0.9
    crit = min(0.97, max(0.5, crit))

    trend = 0.0
    react = 0.35

    if st["zero_frac"] > 0.45:
        # intermittent: sparse positive lumps -- take the empirical quantile of
        # the raw (mostly-zero) history directly.
        q = quantile(hist, crit)
        level0 = q
        react = 0.30
        level = [level0] * period
    elif st["cv2"] > 1.5:
        # bursty: heavy-tailed history dominated by rare huge spikes -- again the
        # empirical quantile of the raw history (not mean+std) is the right
        # single-period newsvendor buffer.
        q = quantile(hist, crit)
        level0 = q
        react = 0.45
        level = [level0] * period
    elif st["seasonal_strength"] > 0.75:
        # seasonal: keep the observed 12-phase shape, lift it by the pooled
        # newsvendor buffer (quantile above the pooled mean).
        q = quantile(hist, crit)
        buf = max(0.0, q - st["mean"])
        level = [st["phase_mean"].get(p, st["mean"]) + buf for p in range(period)]
        react = 0.20
    elif st["r2"] > 0.5:
        # trending: extrapolate the fitted line, buffer with the residual
        # distribution's quantile (noise around the trend, not the raw level).
        resid = [hist[t] - (st["intercept"] + st["slope"] * t) for t in range(n)]
        rq = quantile(resid, crit)
        level0 = st["intercept"] + st["slope"] * (n - 1) + max(0.0, rq)
        trend = st["slope"]
        react = 0.30
        level = [level0] * period
    else:
        # flat / high-variance / looked-calm-but-may-shift: raw-history quantile
        # plus the always-on hedge is the main defense here.
        q = quantile(hist, crit)
        level0 = q
        react = 0.40
        level = [level0] * period

    policies.append({
        "trace_id": tr["trace_id"],
        "level": level,
        "trend": trend,
        "react": react,
    })

print(json.dumps({"policies": policies}))
