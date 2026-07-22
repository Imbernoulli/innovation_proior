import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    out_txt = open(sys.argv[2]).read()

    try:
        it = iter(inp)
        T = int(next(it))
        dt = float(next(it))
        Rmax = float(next(it))
        Fee = float(next(it))
        D = float(next(it))
        r_min = float(next(it))
        prices = [float(next(it)) for _ in range(T)]
        alpha = [float(next(it)) for _ in range(T)]
        E_target = float(next(it))
    except Exception:
        fail("bad input")

    # ---- parse participant artifact: exactly T finite rate tokens ----
    toks = out_txt.split()
    if len(toks) != T:
        fail("expected %d rate values, got %d" % (T, len(toks)))
    rates = []
    for tok in toks:
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite value %r" % tok)
        rates.append(v)

    # ---- feasibility: each rate is EITHER off (~0) OR in [r_min, Rmax] ----
    # A charger cannot idle at an arbitrary trickle: it is off, or it is
    # drawing at least r_min. The "off" tolerance is many orders of
    # magnitude below r_min, so there is a wide dead zone a submission
    # cannot exploit to "bridge" two sessions for near-zero cost -- landing
    # in that dead zone is a hard feasibility violation, not a cheap trick.
    off_tol = 1e-6
    clean_rates = []
    for v in rates:
        if v <= off_tol:
            if v < -1e-6:
                fail("negative rate %.9f" % v)
            clean_rates.append(0.0)
        elif v < r_min - 1e-6:
            fail("rate %.9f is below the minimum operating rate %.6f and above off-tolerance "
                 "%.1e -- a charger must be off or draw >= r_min" % (v, r_min, off_tol))
        elif v > Rmax + 1e-6:
            fail("rate %.6f exceeds Rmax %.3f" % (v, Rmax))
        else:
            clean_rates.append(min(v, Rmax))
    rates = clean_rates

    delivered = sum(v * dt for v in rates)
    energy_eps = max(1e-4, 1e-6 * E_target)
    if delivered < E_target - energy_eps:
        fail("energy target missed: delivered %.4f < target %.4f" % (delivered, E_target))

    def sessions_and_peak(r):
        n_sess = 0
        active = False
        peak = 0.0
        for v in r:
            if v > 0.0:
                if not active:
                    n_sess += 1
                    active = True
                peak = max(peak, v)
            else:
                active = False
        return n_sess, peak

    def bill(r):
        energy_cost = sum(p * v * dt for p, v in zip(prices, r))
        loss_cost = sum(a * v * v * dt for a, v in zip(alpha, r))
        n_sess, peak = sessions_and_peak(r)
        return energy_cost + loss_cost + Fee * n_sess + D * peak

    F_val = bill(rates)

    # ---- internal baseline B: one flat overnight session at the average rate ----
    r_avg = E_target / (T * dt)
    baseline_rates = [r_avg] * T
    B = bill(baseline_rates)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F_val))
    print("bill=%.6f baseline=%.6f Ratio: %.6f" % (F_val, B, sc / 1000.0))


if __name__ == "__main__":
    main()
