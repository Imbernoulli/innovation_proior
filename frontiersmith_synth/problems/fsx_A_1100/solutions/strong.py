# TIER: strong
# The insight (cross-rig consistency): DON'T regress first.  A candidate
# conserved quantity can be tested for CONSTANCY on every rig's collisions
# before any prediction is attempted -- so search invariant space, not
# regression space.
#   1. Grid-search the shared exponent `a` that makes
#      q(a) = m1^a * v1 + m2^a * v2  conserved (pre == post) across ALL rigs'
#      rows simultaneously.  A black-box regressor never asks this question.
#   2. With the invariant fixed, the collision map reduces to one unknown per
#      rig: the restitution e_r = -(v2' - v1')/(v2 - v1), estimated by least
#      squares on relative velocities.
#   3. Multi-experiment consistency: the five rigs give five independent
#      estimates of e; fit ln e_r = ln e0 - beta * gamma_r by weighted least
#      squares so the law (not any single rig) transfers to the unseen rig.
#   4. Emit the closed-form two-body map with m**a and e0*exp(-beta*g).
# What strong does NOT model (headroom): the restitution's residual
# dependence on impact speed |v2 - v1|; a solver that finds it too pushes
# held-out MSE down toward the sensor-noise floor.
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("V1 v1\nV2 v2")
        return
    rigs = []
    rows = []
    i = 3
    while i < len(data):
        if data[i] == "RIG":
            rigs.append(tuple(float(data[i + j]) for j in (1, 2, 3, 4)))
            i += 5
        elif data[i] == "ROW":
            rows.append((int(data[i + 1]),) + tuple(float(data[i + j]) for j in (2, 3, 4, 5)))
            i += 6
        else:
            i += 1
    if not rows:
        print("V1 v1\nV2 v2")
        return

    # ---- 1. invariant search: shared exponent a minimising pooled imbalance ----
    def imbalance(a):
        s = 0.0
        for (ri, v1, v2, y1, y2) in rows:
            m1, m2, g, dt = rigs[ri]
            a1 = m1 ** a
            a2 = m2 ** a
            r = a1 * (y1 - v1) + a2 * (y2 - v2)
            s += r * r
        return s

    best_a, best_v = 1.0, None
    for k in range(206):
        a = 0.60 + 0.004 * k
        v = imbalance(a)
        if best_v is None or v < best_v:
            best_v, best_a = v, a
    # parabolic refine around the grid minimum
    h = 0.004
    v0 = imbalance(best_a)
    vp = imbalance(best_a + h)
    vm = imbalance(best_a - h)
    den = vp - 2.0 * v0 + vm
    if den > 1e-18:
        a_ref = best_a + 0.5 * h * (vm - vp) / den
        if 0.55 < a_ref < 1.45 and imbalance(a_ref) <= v0:
            best_a = a_ref

    # ---- 2. per-rig restitution from relative velocities ----
    # post_rel = (y2 - y1) = -e * (v2 - v1)  =>  e_r = -sum(w * post_rel)/sum(w^2)
    sums = {}
    for (ri, v1, v2, y1, y2) in rows:
        w = v2 - v1
        pr = y2 - y1
        s = sums.setdefault(ri, [0.0, 0.0])
        s[0] += w * pr
        s[1] += w * w
    gam = []
    lne = []
    wts = []
    for ri, (num, den) in sums.items():
        if den <= 1e-9:
            continue
        e_r = -num / den
        if e_r <= 1e-6:
            continue
        gam.append(rigs[ri][2])
        lne.append(math.log(e_r))
        wts.append(den)

    if len(gam) >= 2:
        # ---- 3. weighted log-linear fit: ln e = ln e0 - beta * gamma ----
        W = sum(wts)
        xbar = sum(w * x for w, x in zip(wts, gam)) / W
        ybar = sum(w * y for w, y in zip(wts, lne)) / W
        sxx = sum(w * (x - xbar) ** 2 for w, x in zip(wts, gam))
        sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(wts, gam, lne))
        slope = sxy / sxx if sxx > 1e-12 else 0.0
        intercept = ybar - slope * xbar
    elif len(gam) == 1:
        slope, intercept = 0.0, lne[0]
    else:
        slope, intercept = 0.0, math.log(0.7)

    e0 = math.exp(max(-20.0, min(20.0, intercept)))

    # ---- 4. emit the recovered closed-form law ----
    print("LET a1 m1 ** %.10f" % best_a)
    print("LET a2 m2 ** %.10f" % best_a)
    print("LET ee %.10f * exp ( %.10f * g )" % (e0, slope))
    print("V1 ( a1 * v1 + a2 * v2 + a2 * ee * ( v2 - v1 ) ) / ( a1 + a2 )")
    print("V2 ( a1 * v1 + a2 * v2 - a1 * ee * ( v2 - v1 ) ) / ( a1 + a2 )")


if __name__ == "__main__":
    main()
