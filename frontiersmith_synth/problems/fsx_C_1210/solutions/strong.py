# TIER: strong
# The insight: staggered entry means several cohorts of DIFFERENT ages are visible
# on the SAME calendar day, and they all feel the exact same common calendar
# wobble that day. Differencing two cohorts' lifts on a shared calendar day makes
# the wobble (and the persistent lift, which is also common) cancel EXACTLY,
# leaving a signal that depends only on the two ages and the decay curve:
#     L(age_i,t) - L(age_j,t) = A*(exp(-age_i/tau) - exp(-age_j/tau)) + noise
# Grid-search tau; for each candidate tau, A is a 1-D linear least squares fit on
# all cross-cohort same-day differences (pooled across every day). Pick the tau
# that minimizes the residual sum of squares. Then strip A*exp(-age/tau) back out
# of every row and average what's left -- the calendar wobble is (near) zero-mean
# across the many days observed, so this recovers the persistent lift P. Emit
# P_hat + A_hat*exp(-age/tau_hat): this is the correct functional SHAPE, so it
# extrapolates to the held-out horizon far beyond the visible window, where naive
# tail-averaging is still measurably biased by the still-decaying novelty term.
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.05"); return
    n = int(data[0])
    vals = data[2:]
    rows = []
    for i in range(n):
        c = int(vals[5 * i]); s = int(vals[5 * i + 1]); t = int(vals[5 * i + 2])
        age = int(vals[5 * i + 3]); L = float(vals[5 * i + 4])
        rows.append((c, s, t, age, L))

    by_day = {}
    for c, s, t, age, L in rows:
        by_day.setdefault(t, []).append((age, L))

    diffs = []  # (age_i, age_j, delta_L)
    for t, lst in by_day.items():
        if len(lst) < 2:
            continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                age_i, L_i = lst[a]
                age_j, L_j = lst[b]
                if age_i == age_j:
                    continue
                diffs.append((age_i, age_j, L_i - L_j))

    def sse_for_tau(tau):
        num = 0.0
        den = 0.0
        bases = []
        for age_i, age_j, dL in diffs:
            basis = math.exp(-age_i / tau) - math.exp(-age_j / tau)
            bases.append(basis)
            num += basis * dL
            den += basis * basis
        A_hat = num / den if den > 1e-12 else 0.0
        sse = 0.0
        for (age_i, age_j, dL), basis in zip(diffs, bases):
            resid = dL - A_hat * basis
            sse += resid * resid
        return sse, A_hat

    best_tau, best_A, best_sse = 6.0, 0.0, float("inf")
    if diffs:
        tau = 1.5
        while tau <= 45.0:
            sse, A_hat = sse_for_tau(tau)
            if sse < best_sse:
                best_sse, best_tau, best_A = sse, tau, A_hat
            tau *= 1.06
    tau_hat = best_tau
    A_hat = best_A

    resid_sum = 0.0
    for c, s, t, age, L in rows:
        resid_sum += L - A_hat * math.exp(-age / tau_hat)
    P_hat = resid_sum / len(rows)

    print("%.8g + %.8g*exp(-age/%.8g)" % (P_hat, A_hat, tau_hat))


if __name__ == "__main__":
    main()
