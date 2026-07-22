import sys, random, math

# ---------------------------------------------------------------------------
# Instance generator for "Depleting the Budget: Retrofit Scheduling Against a
# Demand Calendar" (family: retrofit-window-sequencing).
#
# U power units, each with a dirty (pre-retrofit) emission rate, a clean
# (post-retrofit) emission rate, and a fixed retrofit duration during which
# the unit is fully offline. A demand calendar over T steps has a seasonal
# peak. A hard-ish cumulative emissions BUDGET (soft-penalized beyond it)
# governs the whole horizon.
#
# TRAP_IDS place the seasonal peak so it deterministically straddles the
# retrofit window that a naive "dirtiest unit first, packed back-to-back
# from t=1" schedule would produce for the SECOND-dirtiest unit -- forcing
# extra dirty must-cover dispatch right when the fleet can least afford it.
# NON-TRAP ids place the peak safely after all naive retrofits would already
# be finished, so the ordering choice alone is not punished there (only the
# missed bonus-dispatch opportunity is).
# ---------------------------------------------------------------------------

TRAP_IDS = {2, 3, 5, 6, 8, 9, 10}


def capacity_and_rate(t, s_i, C_i, R_i, r_i, D_i):
    """t is 1-indexed. s_i=0 means never retrofit."""
    if s_i == 0:
        return C_i, R_i
    off_start, off_end = s_i, s_i + D_i - 1
    if off_start <= t <= off_end:
        return 0.0, 0.0
    if t < off_start:
        return C_i, R_i
    return C_i, r_i


def naive_baseline_emissions(U, T, C, R, demand):
    """Never retrofit; fixed INDEX-order fill to exactly meet demand each
    step (no bonus). Same construction the checker uses internally."""
    total_em = 0.0
    for t in range(T):
        need = demand[t]
        for i in range(U):
            if need <= 1e-9:
                break
            use = min(C[i], need)
            total_em += use * R[i]
            need -= use
    return total_em


def reference_schedule(U, T, C, R, r, D, demand):
    """A demand-calendar-aware retrofit schedule: process units in
    descending cleanup payoff (R-r)*C, and for each place its offline
    window at the EARLIEST start that causes zero projected coverage
    shortfall against everyone already committed offline (falling back to
    the least-bad option). This is the reference used ONLY to calibrate a
    realistic, achievable BUDGET for the instance -- it deliberately mirrors
    the insight a strong solver should find, not a hard-coded answer."""
    total_cap = sum(C)
    order = sorted(range(U), key=lambda i: -(R[i] - r[i]) * C[i])
    sched = [0] * U
    offline_cap = [0.0] * (T + 2)
    for i in order:
        best = None
        for s in range(1, T - D[i] + 2):
            shortfall = 0.0
            for t in range(s, s + D[i]):
                avail = total_cap - C[i] - offline_cap[t]
                if avail < demand[t - 1]:
                    shortfall += demand[t - 1] - avail
            key = (shortfall, s)
            if best is None or key < best:
                best = key
        s_best = best[1]
        sched[i] = s_best
        for t in range(s_best, s_best + D[i]):
            offline_cap[t] += C[i]
    return sched


def mandatory_emissions(U, T, C, R, r, D, schedule, demand):
    """Merit-order (cheapest-rate-first) dispatch to exactly meet demand
    each step under a given retrofit schedule -- no bonus dispatch."""
    em = 0.0
    for t in range(1, T + 1):
        specs = []
        for i in range(U):
            cap, rate = capacity_and_rate(t, schedule[i], C[i], R[i], r[i], D[i])
            if cap > 1e-9:
                specs.append((rate, cap))
        specs.sort()
        need = demand[t - 1]
        for rate, cap in specs:
            if need <= 1e-9:
                break
            use = min(cap, need)
            em += use * rate
            need -= use
    return em


def build_instance(test_id):
    rng = random.Random(2026000 + test_id * 7919)
    U = 5 + (test_id - 1) // 2          # 5..9
    T = 34 + 4 * (test_id - 1)          # 34..70
    is_trap = test_id in TRAP_IDS

    # ---- rank-ordered specs: rank 0 = dirtiest ----
    R_MAX = 9.0
    R_STEP = 8.0 / max(1, (U - 1))
    R = [round(R_MAX - rk * R_STEP + rng.uniform(-0.12, 0.12), 2) for rk in range(U)]
    r_clean = [round(max(0.5, R[rk] * 0.22 + rng.uniform(-0.04, 0.04)), 2) for rk in range(U)]
    C = [12.0 + rng.randint(-2, 3) for _ in range(U)]
    D = [5 + (rk % 4) + rng.randint(0, 1) for rk in range(U)]
    D = [int(min(d, max(3, T // 5))) for d in D]

    total_cap = sum(C)

    # ---- demand calendar: trapezoid seasonal peak ----
    floor_frac = 0.38
    peak_frac = 0.74
    base = floor_frac * total_cap
    peak = peak_frac * total_cap

    if is_trap:
        cursor = 1 + D[0]                      # where rank-1 unit starts under naive packing
        peak_lo = cursor + max(1, D[1] // 3)
        peak_hi = peak_lo + max(D[1], T // 6)
        peak_hi = min(peak_hi, T - 2)
        peak_lo = max(2, min(peak_lo, peak_hi - 2))
    else:
        total_pack = sum(D)                    # naive schedule finishes retrofitting everyone by here
        peak_lo = min(T - 6, total_pack + max(3, T // 8))
        peak_hi = min(T - 2, peak_lo + max(4, T // 8))
        if peak_lo >= peak_hi:
            peak_lo, peak_hi = max(2, T - 6), T - 2

    ramp = max(2, (peak_hi - peak_lo) // 3)
    demand = []
    for t in range(1, T + 1):
        if t < peak_lo:
            d = base
        elif t <= peak_lo + ramp:
            frac = (t - peak_lo) / ramp
            d = base + frac * (peak - base)
        elif t <= peak_hi - ramp:
            d = peak
        elif t <= peak_hi:
            frac = (peak_hi - t) / ramp
            d = base + frac * (peak - base)
        else:
            d = base
        d += rng.uniform(-0.03, 0.03) * base
        demand.append(round(max(1.0, d), 3))

    # ---- scramble presentation order (unit identity independent of dirtiness rank) ----
    perm = list(range(U))
    rng.shuffle(perm)
    C2 = [C[perm[i]] for i in range(U)]
    R2 = [R[perm[i]] for i in range(U)]
    r2 = [r_clean[perm[i]] for i in range(U)]
    D2 = [D[perm[i]] for i in range(U)]

    # ---- calibrate BUDGET from a demand-calendar-aware REFERENCE schedule's
    # own mandatory (no-bonus) emissions -- tight for trap cases (so a bad,
    # calendar-blind schedule's extra crunch emissions genuinely blow it),
    # generous for control cases (so scheduling quality is not the binding
    # constraint there -- only the missed bonus-dispatch upside is). ----
    ref_sched = reference_schedule(U, T, C2, R2, r2, D2, demand)
    ref_em = mandatory_emissions(U, T, C2, R2, r2, D2, ref_sched, demand)
    factor = 1.3 if is_trap else 2.2
    BUDGET = round(factor * ref_em, 3)
    PEN = 0.16

    return U, T, PEN, BUDGET, C2, R2, r2, D2, demand


def main():
    test_id = int(sys.argv[1])
    U, T, PEN, BUDGET, C, R, r, D, demand = build_instance(test_id)

    out = [f"{U} {T}", f"{PEN:.4f} {BUDGET:.4f}"]
    for i in range(U):
        out.append(f"{C[i]:.3f} {R[i]:.3f} {r[i]:.3f} {D[i]}")
    out.append(" ".join(f"{x:.3f}" for x in demand))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
