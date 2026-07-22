# TIER: strong
# The insight: the market rhythm and the scribe cap are FIXED across the
# decree -- only the growth rate changes. So identify them separately
# instead of fitting the reported curve as one blob:
#
#  1. RECOVER THE GROWTH RATE FROM AN INVARIANT. For a fixed weekday d, the
#     load is X0*w[d]*exp(beta*t): the SAME multiplicative weekly factor w[d]
#     multiplies every occurrence of that weekday, so it cancels out of the
#     ratio between two EARLY (unsaturated) occurrences of that weekday:
#         reported(t2)/reported(t1) ~= exp(beta*(t2-t1))   when far from cap.
#     Averaging this ratio (as a log-slope) over all 7 weekdays, using only
#     the first two occurrences of each (guaranteed unsaturated by
#     construction), gives a beta estimate immune to BOTH the rhythm and the
#     saturation -- neither nuisance parameter ever enters the estimate.
#  2. RECOVER THE REPORTING OPERATOR LINEARLY. Given beta, invert the
#     saturating link: 1/r(t) = 1/Cap + (1/(X0*w[d])) * exp(-beta*t). This is
#     LINEAR in (1/Cap, 1/(X0*w[d])) for each weekday d, so an ordinary
#     per-weekday least-squares fit of 1/r against exp(-beta*t) recovers a
#     level L0[d] = X0*w[d] and a shared cap estimate.
#  3. EXTRAPOLATE. Only the growth rate is decree-sensitive: post-decree load
#     is L0[dow(t)] * exp(beta*n) * exp(beta*f*(t-n)); push it back through
#     the SAME (unchanged) saturating link -- never through the raw trend.
import sys, math


def median(xs):
    xs = sorted(xs)
    m = len(xs)
    if m == 0:
        return 0.08
    if m % 2 == 1:
        return xs[m // 2]
    return 0.5 * (xs[m // 2 - 1] + xs[m // 2])


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("0.0"); return
    n = int(data[0])
    rows = data[3:]
    t_list = [int(rows[2 * i]) for i in range(n)]
    r_list = [float(rows[2 * i + 1]) for i in range(n)]

    by_dow = {d: [] for d in range(7)}
    for t, r in zip(t_list, r_list):
        by_dow[t % 7].append((t, r))
    for d in by_dow:
        by_dow[d].sort()

    # --- step 1: growth rate from early-week same-weekday ratios ---
    local_betas = []
    for d in range(7):
        seq = by_dow[d]
        if len(seq) >= 2:
            (t0, r0), (t1, r1) = seq[0], seq[1]
            if r0 > 0.5 and t1 > t0:
                local_betas.append(math.log(max(r1, 0.5) / r0) / (t1 - t0))
    beta_hat = median(local_betas) if local_betas else 0.08
    beta_hat = max(0.01, min(0.5, beta_hat))

    # --- step 2: per-weekday linear fit of 1/r against exp(-beta*t) ---
    A = {}   # slope ~ 1/(X0*w[d])
    C0 = []  # intercepts ~ 1/Cap
    for d in range(7):
        seq = by_dow[d]
        if len(seq) < 2:
            continue
        xs = [math.exp(-beta_hat * t) for t, _ in seq]
        ys = [1.0 / max(r, 0.5) for _, r in seq]
        m = len(seq)
        sx = sum(xs); sy = sum(ys)
        sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
        denom = m * sxx - sx * sx
        if abs(denom) < 1e-12:
            continue
        slope = (m * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / m
        if slope > 1e-9:
            A[d] = slope
        C0.append(intercept)

    C0_bar = sum(C0) / len(C0) if C0 else 1e-6
    Cap_hat = 1.0 / C0_bar if C0_bar > 1e-9 else 1e6
    Cap_hat = max(Cap_hat, 1.0)

    valid_L0 = {d: 1.0 / A[d] for d in A}
    default_L0 = (sum(valid_L0.values()) / len(valid_L0)) if valid_L0 else 5.0
    L0 = [valid_L0.get(d, default_L0) for d in range(7)]

    L0_str = "[ " + " , ".join("%.8f" % v for v in L0) + " ]"
    load = "( ( %s ) [ t %% 7 ] * exp ( %.10f * n ) * exp ( %.10f * f * ( t - n ) ) )" % (
        L0_str, beta_hat, beta_hat)
    expr = "( %.10f * %s ) / ( %.10f + %s )" % (Cap_hat, load, Cap_hat, load)
    print(expr)


if __name__ == "__main__":
    main()
