# TIER: strong
# The insight: the calm log's (e, I) -> u map already carries a faint but
# real THIRD-ORDER curvature (the Taylor tail of a tanh compression) even
# though the throttle never actually saturates during calm flight.  Rather
# than fit a linear PI law (the greedy trap), COMMIT to the structurally
# right family -- a throttle with a smooth, symmetric authority limit --
# and recover its scale from that curvature:
#
#   u = Umax * tanh( (Kp*e + Ki*I) / Umax )
#
# For a FIXED candidate Umax, atanh(u/Umax)*Umax is *linear* in (e, I), so
# each candidate reduces to a closed-form least-squares fit; we grid-search
# Umax (coarse, then refined) and keep the candidate with the lowest
# training error in u-space.  This generalises to the stormy held-out
# flight, where the real throttle's limited authority (and the windup that
# authority limit produces) dominates the response.
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("OUT 0.0 * e")
        return
    n = int(data[0])
    vals = data[3:]
    es = []
    us = []
    Is = []
    I = 0.0
    for i in range(n):
        sp = float(vals[3 * i])
        y = float(vals[3 * i + 1])
        u = float(vals[3 * i + 2])
        e = sp - y
        I = I + e
        es.append(e)
        Is.append(I)
        us.append(u)

    See = sum(e * e for e in es)
    SeI = sum(e * Iv for e, Iv in zip(es, Is))
    SII = sum(Iv * Iv for Iv in Is)
    det0 = See * SII - SeI * SeI

    def fit_linear(zs):
        Sez = sum(e * z for e, z in zip(es, zs))
        SIz = sum(Iv * z for Iv, z in zip(Is, zs))
        if abs(det0) < 1e-9:
            return (Sez / See if See > 1e-9 else 0.0), 0.0
        Kp = (Sez * SII - SeI * SIz) / det0
        Ki = (See * SIz - SeI * Sez) / det0
        return Kp, Ki

    def train_mse(Kp, Ki, Um):
        s = 0.0
        for e, Iv, uv in zip(es, Is, us):
            pred = Um * math.tanh((Kp * e + Ki * Iv) / Um)
            s += (pred - uv) ** 2
        return s / n

    def candidate(Um):
        zs = [math.atanh(max(-0.995, min(0.995, uv / Um))) * Um for uv in us]
        Kp, Ki = fit_linear(zs)
        return Kp, Ki, train_mse(Kp, Ki, Um)

    # Search cap: a lazy throttle has SOME finite authority limit. We search
    # a generous but FIXED range and never let refinement drift past it --
    # walking the cap outward every round is exactly how you'd talk yourself
    # into "no saturation" (Umax -> infinity), the greedy trap.
    UMAX_CAP = 12.0
    UMAX_LO = 0.3
    best = None
    for i in range(80):
        Um = UMAX_LO + (UMAX_CAP - UMAX_LO) * i / 79.0
        Kp, Ki, mse = candidate(Um)
        if best is None or mse < best[0]:
            best = (mse, Kp, Ki, Um)

    mse, Kp, Ki, Umax = best
    lo, hi = max(UMAX_LO, Umax * 0.7), min(UMAX_CAP, Umax * 1.4)
    for _ in range(4):
        for i in range(30):
            Um = lo + (hi - lo) * i / 29.0
            if Um < UMAX_LO:
                continue
            Kp2, Ki2, mse2 = candidate(Um)
            if mse2 < mse:
                mse, Kp, Ki, Umax = mse2, Kp2, Ki2, Um
        lo, hi = max(UMAX_LO, Umax * 0.85), min(UMAX_CAP, Umax * 1.18)

    print("OUT %.6f * tanh ( ( %.6f * e + %.6f * I ) / %.6f )" % (Umax, Kp, Ki, Umax))


if __name__ == "__main__":
    main()
