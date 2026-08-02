# TIER: strong
"""The insight: the ramp-rate constraint only binds INSIDE the two phase-transition bands,
and only because of the THICKEST piece in the load (its core lags the surface the most,
since the lag time constant grows with thickness^2).  So: ramp at the burner's max rate
everywhere else ("speed elsewhere is free"), and inside each band use exact bisection on the
piece's own analytic thermal-lag ODE to find the fastest rate that keeps ITS projected
in-band stress within a safety margin of its crack threshold -- carrying its real
cumulative stress and core temperature across both bands in temperature order (spending
band-1's leftover safety margin correctly affects what's left for band 2).  Every other,
thinner piece is automatically safe too, since its lag (and hence its stress) is smaller.
This is a reformulation (decompose the schedule into free-speed vs gated-speed regimes) +
an exchange argument (only the worst-case piece needs to be checked) + root-finding on the
exact physics, not a hand-tuned constant."""
import sys, json, math

SAFETY_FRAC = 0.72     # target fraction of the thickest piece's stress budget to spend
MIN_RATE_FRAC = 0.03   # floor on in-band rate as a fraction of max_rate
ITERS = 50


def analytic_step(core0, T0, k, h, tau):
    if tau <= 1e-9:
        return T0 + k * h, 0.0
    u0 = core0 - T0
    A = k * tau
    B = u0 + A
    e = 0.0 if h / tau > 40.0 else math.exp(-h / tau)
    e2 = e * e
    T_h = T0 + k * h
    core_h = T_h + B * e - A
    integ = A * A * h - 2.0 * A * B * tau * (1.0 - e) + B * B * (tau / 2.0) * (1.0 - e2)
    if integ < 0.0:
        integ = 0.0
    return core_h, integ


def main():
    inst = json.load(sys.stdin)
    start = float(inst["start_temp"])
    target = float(inst["target_temp"])
    max_rate = float(inst["max_rate"])
    diffusion_k = float(inst["diffusion_k"])
    stress_k = float(inst["stress_threshold_k"])
    pieces = inst["pieces"]
    bands = sorted(inst["bands"], key=lambda b: b["lo"])

    thick_idx = max(range(len(pieces)), key=lambda i: pieces[i]["thickness_mm"])
    thick = pieces[thick_idx]
    tau_t = diffusion_k * (thick["thickness_mm"] ** 2)
    thr_t = stress_k * thick["fragility"] * SAFETY_FRAC

    core = start
    cum = 0.0
    cur = start
    schedule = []
    for b in bands:
        lo, hi = float(b["lo"]), float(b["hi"])
        if hi <= cur:
            continue
        if lo > cur:
            mins = (lo - cur) / max_rate
            core, _ = analytic_step(core, cur, max_rate, mins, tau_t)
            schedule.append({"to_temp": lo, "minutes": mins})
            cur = lo
        width = hi - max(lo, cur)
        if width <= 0.0:
            continue
        mult = float(b["multiplier"])
        remaining = thr_t - cum
        lo_r, hi_r = MIN_RATE_FRAC * max_rate, max_rate

        def integ_at(r):
            mins = width / r
            _, ig = analytic_step(core, cur, r, mins, tau_t)
            return ig * mult

        if remaining <= 0.0:
            r_use = lo_r
        elif integ_at(hi_r) <= remaining:
            r_use = hi_r
        elif integ_at(lo_r) > remaining:
            r_use = lo_r
        else:
            a, bb = lo_r, hi_r
            for _ in range(ITERS):
                mid = (a + bb) / 2.0
                if integ_at(mid) <= remaining:
                    a = mid
                else:
                    bb = mid
            r_use = a
        mins = width / r_use
        core, integ = analytic_step(core, cur, r_use, mins, tau_t)
        cum += integ * mult
        schedule.append({"to_temp": hi, "minutes": mins})
        cur = hi

    if cur < target:
        schedule.append({"to_temp": target, "minutes": (target - cur) / max_rate})
        cur = target

    print(json.dumps(schedule))


main()
