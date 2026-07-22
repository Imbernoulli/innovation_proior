# TIER: strong
import sys, bisect


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    T = int(next(it))
    K, A, Acol, gamma, p_base, Qcap = (float(next(it)) for _ in range(6))
    S0, r_base = (float(next(it)) for _ in range(2))
    dstart = int(next(it)); dend = int(next(it)); drought_mult = float(next(it))
    n_closed = int(next(it))
    closed_mod = set(int(next(it)) for _ in range(n_closed))
    n_season = int(next(it))
    price_season = [float(next(it)) for _ in range(n_season)]
    N = int(next(it))
    costs = []
    caps = []
    for _ in range(N):
        c = float(next(it)); cap = float(next(it))
        costs.append(c); caps.append(cap)

    order = sorted(range(N), key=lambda idx: costs[idx])
    sorted_costs = [costs[idx] for idx in order]
    prefix = [0.0]
    for idx in order:
        prefix.append(prefix[-1] + caps[idx])

    def attempted_catch(thresh):
        j = bisect.bisect_left(sorted_costs, thresh)
        return prefix[j]

    def build_quotas(cyc_open, cyc_closed, burst_frac, safety_buf):
        # Insight: under the depensation cubic, a fixed constant quota either wastes
        # headroom (too low) or drags the stock through the low-growth/collapse zone
        # the moment the printed drought window hits (too high) -- because a constant
        # drain never lets the stock compound back into the fast-growth interior.
        # Instead: ALWAYS close during the legal season and the printed drought
        # window (voluntarily, before any damage happens), and outside those windows
        # alternate short bursts (harvest near the regulatory ceiling while stock is
        # comfortably above the Allee threshold) with short closures that let the
        # stock regrow. This pulsing keeps weekly catch mostly quota-capped (the
        # binding constraint), rather than effort-capped or growth-starved.
        cyc = cyc_open + cyc_closed
        S = S0
        quotas = [0.0] * T
        for t in range(T):
            wk = t % 52
            in_drought = dstart <= t < dend
            r_t = r_base * (drought_mult if in_drought else 1.0)
            growth = r_t * S * (S / A - 1.0) * (1.0 - S / K)
            Sp = S + growth
            if Sp < 0.0:
                Sp = 0.0
            Q = 0.0
            if not (wk in closed_mod or in_drought):
                phase = t % cyc
                if phase < cyc_open and Sp > safety_buf * A:
                    Q = burst_frac * Qcap
            Q = max(0.0, min(Q, Qcap))
            quotas[t] = Q
            p_t = p_base * price_season[wk]
            thresh = p_t * (Sp / K)
            attempt = attempted_catch(thresh)
            H = min(Q, attempt, Sp)
            S = Sp - H
        return quotas

    def simulate(quotas):
        S = S0
        total = 0.0
        disc = 1.0
        for t in range(T):
            wk = t % 52
            Q = quotas[t]
            r_t = r_base * (drought_mult if dstart <= t < dend else 1.0)
            growth = r_t * S * (S / A - 1.0) * (1.0 - S / K)
            Sp = S + growth
            if Sp < 0.0:
                Sp = 0.0
            p_t = p_base * price_season[wk]
            thresh = p_t * (Sp / K)
            attempt = attempted_catch(thresh)
            H = min(Q, attempt, Sp)
            Snext = Sp - H
            if Snext < Acol - 1e-6:
                return None
            total += disc * p_t * H
            disc *= gamma
            S = Snext
        return total

    best_val = -1.0
    best_q = None
    for cyc_open in (4, 8, 13, 20, 26):
        for cyc_closed in (2, 4, 8):
            for burst_frac in (0.3, 0.5, 0.7, 0.9, 1.0):
                for safety_buf in (1.05, 1.15, 1.3):
                    q = build_quotas(cyc_open, cyc_closed, burst_frac, safety_buf)
                    v = simulate(q)
                    if v is not None and v > best_val:
                        best_val = v
                        best_q = q

    if best_q is None:
        # ultra-conservative fallback: never fish (always feasible, zero revenue)
        best_q = [0.0] * T

    sys.stdout.write("\n".join("%.6f" % q for q in best_q) + "\n")


if __name__ == "__main__":
    main()
