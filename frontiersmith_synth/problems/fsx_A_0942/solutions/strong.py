# TIER: strong
# The insight: the individual delivery TIMES (and whether a day's total was
# one delivery or two) are unobserved and cannot be recovered -- but they
# don't need to be. For the TRUE (noise-free) levels, mass balance over a
# full day is exact regardless of WHEN within the day deliveries landed:
#     decay_lost_that_day = Q_d + E_(d-1) - E_d
# This holds whether the delivery happened at dawn, dusk, or split across
# two unlogged sub-deliveries -- the accounting identity absorbs the
# unobserved timing/count completely, on EVERY day (not just the
# delivery-free ones), so no training data is wasted filtering "suspicious"
# days. Computed from the NOISY telemetry we actually get, decay_lost is an
# unbiased (not exact) estimate of that day's leakage -- still far better
# than ignoring Q_d. log(decay_lost/T_day) vs log(level) then gives a clean
# regression for the leak exponent alpha and constant c -- weighted by
# decay_lost^2 (an inverse-variance-style weight) so days with a tiny,
# noise-dominated decay signal don't distort the fit. Notice we never
# estimate a single event time anywhere; the conservation identity is the
# whole trick.
import sys, math

T_DAY = 1.0


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.001*L")
        return
    D = int(data[0])
    L0 = float(data[2])
    rows = []
    idx = 3
    for _ in range(D):
        Q = float(data[idx]); E = float(data[idx + 1]); idx += 2
        rows.append((Q, E))

    xs, ys, ws = [], [], []
    E_prev = L0
    for Q, E in rows:
        decay_amt = Q + E_prev - E          # mass-balance identity
        Lref = (E_prev + Q / 2.0 + E) / 2.0  # representative level for the day
        if decay_amt > 1e-6 and Lref > 1e-6:
            xs.append(math.log(Lref))
            ys.append(math.log(decay_amt / T_DAY))
            ws.append(decay_amt * decay_amt)
        E_prev = E

    if len(xs) < 3:
        print("0.001*L")
        return

    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    if sxx < 1e-9:
        slope, intercept = 0.0, my
    else:
        slope = sxy / sxx
        intercept = my - slope * mx

    c = math.exp(intercept)
    print("%.8f * L ** %.6f" % (c, slope))


if __name__ == "__main__":
    main()
