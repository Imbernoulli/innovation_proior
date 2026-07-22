# TIER: strong
# Two-layer insight over the same combinatorial-plus-continuous structure:
#
# 1) SUPPORT SELECTION (discrete). Session fees make the choice of which
#    steps to ever charge from a combinatorial decision: pulling in one more
#    cheap-but-distant step can merge two sessions into one (saving a fee) or
#    open a brand new one (costing a fee), and it changes how much quadratic
#    loss / demand charge the remaining steps must carry. We sweep the
#    support S over "the k cheapest steps by price", k = 1..T -- exactly the
#    sequence of supports a rising water level ever activates -- and for each
#    S solve the continuous sub-problem that ignores the demand charge:
#    minimize sum(p*r + alpha*r^2)*dt s.t. sum(r*dt) = E_target, 0<=r<=Rmax,
#    which is water-filling -- price plus marginal quadratic loss equalized
#    at a level lambda found by bisection. We price the REALIZED cost
#    (including the demand charge and the actual positive-rate session
#    pattern -- water-filling can zero out an included step) and keep the
#    cheapest k.
#
# 2) PEAK SHAPING (continuous, for the winning support). The demand charge
#    D*max(r) is NOT part of the water-filling objective above, so step 1
#    alone under-weights it. Given the winning support, we additionally
#    search over an operating CEILING cap <= Rmax for that support (a
#    ternary search, since lowering the cap trades a higher, more spread-out
#    water level -- more quadratic loss -- against a lower demand charge).
#    This is the second half of the insight: not just which slots to use,
#    but how hard to let any one of them run.
#
# A hardware floor also applies: a charger is off, or it draws at least
# r_min; we snap any water-filled trickle below r_min up to r_min (this can
# only add delivered energy, never break the energy-target feasibility gate).
import sys


def water_fill(idx, prices, alpha, cap, dt, E_target):
    """Rates on `idx` (others 0) minimizing sum(p*r+alpha*r^2)*dt s.t.
    sum(r*dt)==E_target, 0<=r<=cap, via bisection on the water level."""
    if not idx:
        return None
    capacity = cap * dt * len(idx)
    if capacity < E_target - 1e-9:
        return None

    def energy(lam):
        tot = 0.0
        for t in idx:
            r = (lam - prices[t]) / (2.0 * alpha[t])
            if r < 0.0:
                r = 0.0
            elif r > cap:
                r = cap
            tot += r * dt
        return tot

    lo = min(prices[t] for t in idx) - 1.0
    hi = max(prices[t] + 2.0 * alpha[t] * cap for t in idx) + 1.0
    for _ in range(60):
        if energy(hi) >= E_target - 1e-9:
            break
        hi += (hi - lo) + 1.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if energy(mid) < E_target - 1e-9:
            lo = mid
        else:
            hi = mid
    lam = hi
    r_by_idx = {}
    for t in idx:
        r = (lam - prices[t]) / (2.0 * alpha[t])
        if r < 0.0:
            r = 0.0
        elif r > cap:
            r = cap
        r_by_idx[t] = r
    tot = sum(r_by_idx[t] * dt for t in idx)
    if tot > 1e-9:
        scale = E_target / tot
        if all(0.0 <= r_by_idx[t] * scale <= cap + 1e-6 for t in idx):
            if abs(tot - E_target) > 1e-6:
                for t in idx:
                    r_by_idx[t] = min(cap, r_by_idx[t] * scale)
    return r_by_idx


def enforce_rmin(rates, r_min):
    """A charger is off (0) or draws >= r_min; snap any trickle up."""
    return [0.0 if v <= 1e-9 else max(v, r_min) for v in rates]


def bill(rates, prices, alpha, Fee, D, dt):
    energy_cost = sum(p * v * dt for p, v in zip(prices, rates))
    loss_cost = sum(a * v * v * dt for a, v in zip(alpha, rates))
    n_sess = 0
    active = False
    peak = 0.0
    for v in rates:
        if v > 1e-9:
            if not active:
                n_sess += 1
                active = True
            peak = max(peak, v)
        else:
            active = False
    return energy_cost + loss_cost + Fee * n_sess + D * peak


def rates_for_support(active, cap, prices, alpha, dt, E_target, T, r_min):
    r_by_idx = water_fill(active, prices, alpha, cap, dt, E_target)
    if r_by_idx is None:
        return None
    rates = [0.0] * T
    for idx_t, v in r_by_idx.items():
        rates[idx_t] = v
    return enforce_rmin(rates, r_min)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    dt = float(next(it))
    Rmax = float(next(it))
    Fee = float(next(it))
    D = float(next(it))
    r_min = float(next(it))
    prices = [float(next(it)) for _ in range(T)]
    alpha = [float(next(it)) for _ in range(T)]
    E_target = float(next(it))

    order = sorted(range(T), key=lambda t: (prices[t], t))

    # ---- layer 1: sweep the support (which price ranks get any energy) ----
    best_cost = None
    best_rates = None
    best_active = None
    active = []
    for k in range(1, T + 1):
        active.append(order[k - 1])
        rates = rates_for_support(active, Rmax, prices, alpha, dt, E_target, T, r_min)
        if rates is None:
            continue
        c = bill(rates, prices, alpha, Fee, D, dt)
        if best_cost is None or c < best_cost:
            best_cost, best_rates, best_active = c, rates, list(active)

    if best_rates is None:
        r_avg = max(E_target / (T * dt), r_min)
        best_rates = [r_avg] * T
        best_active = list(range(T))
        best_cost = bill(best_rates, prices, alpha, Fee, D, dt)

    # ---- layer 2: for the winning support, shape the peak against D ----
    n_active = len(best_active)
    cap_lo = E_target / (dt * n_active)   # tightest cap that can still meet target
    cap_hi = max(best_rates) if max(best_rates) > cap_lo else Rmax
    cap_hi = min(Rmax, max(cap_hi, cap_lo))
    if cap_hi - cap_lo > 1e-6:
        lo, hi = cap_lo, cap_hi
        for _ in range(40):
            x = lo + (hi - lo) / 3.0
            y = hi - (hi - lo) / 3.0
            rx = rates_for_support(best_active, x, prices, alpha, dt, E_target, T, r_min)
            ry = rates_for_support(best_active, y, prices, alpha, dt, E_target, T, r_min)
            cx = bill(rx, prices, alpha, Fee, D, dt) if rx is not None else float("inf")
            cy = bill(ry, prices, alpha, Fee, D, dt) if ry is not None else float("inf")
            if cx < cy:
                hi = y
            else:
                lo = x
        refined = rates_for_support(best_active, (lo + hi) / 2.0, prices, alpha, dt, E_target, T, r_min)
        if refined is not None:
            rc = bill(refined, prices, alpha, Fee, D, dt)
            if rc < best_cost:
                best_cost, best_rates = rc, refined

    print(" ".join(f"{r:.6f}" for r in best_rates))


if __name__ == "__main__":
    main()
