# TIER: strong
# The insight: each training ticket's pass/fail bit is NOT a classification label
# to curve-fit -- it is an INEQUALITY CONSTRAINT on the two latent rate constants
# (k_heat for heating, k_cool for cooling). Treat the whole training set as a
# constraint-propagation problem: jointly search (k_heat, k_cool) over a WIDE
# log-spaced grid (coarse-to-fine, refining around the best region each round)
# for the pair CONSISTENT with the most training tickets -- correctly composing
# the ramp-then-hold trajectory with whichever rate applies to each segment's
# own direction, rather than assuming every hold starts from ambient. Once
# (k_heat, k_cool) is pinned down, any held-out schedule -- however short,
# however far its setpoints extrapolate beyond the training range -- is
# answered by exactly re-simulating the same closed-form two-rate law. This is
# what lets it generalize across the regime-extrapolation split where a recipe
# that only memorizes feature/label correlation, or a data-blind guess at a
# "typical" rate, cannot.
import math
import sys

T0 = 20.0


def seg_end(S, Tstart, k, D):
    return S + (Tstart - S) * math.exp(-k * D)


def rate_for(S, Tstart, k_h, k_c):
    return k_h if S >= Tstart else k_c


def eval_job(S1, D1, S2, D2, w, Lo, Hi, k_h, k_c):
    if D1 > 0:
        k1 = rate_for(S1, T0, k_h, k_c)
        T_mid = seg_end(S1, T0, k1, D1)
    else:
        T_mid = T0
    k2 = rate_for(S2, T_mid, k_h, k_c)
    checkStart = D2 - w
    Ta = seg_end(S2, T_mid, k2, checkStart)
    Tb = seg_end(S2, T_mid, k2, D2)
    lo_t, hi_t = (Ta, Tb) if Ta <= Tb else (Tb, Ta)
    margin = min(lo_t - Lo, Hi - hi_t)
    nm = margin / ((Hi - Lo) / 2.0)
    label = 1 if margin >= 0.0 else 0
    return nm, label


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    testId = int(next(it)); N = int(next(it)); Q = int(next(it))
    train = []
    for _ in range(N):
        vals = [float(next(it)) for _ in range(7)]
        label = int(next(it))
        train.append(tuple(vals) + (label,))
    query = []
    for _ in range(Q):
        vals = [float(next(it)) for _ in range(7)]
        query.append(tuple(vals))

    kh_lo, kh_hi = 0.005, 2.0
    kc_lo, kc_hi = 0.001, 1.0
    best = None
    STEPS = 20
    for stage in range(5):
        rh = kh_hi / kh_lo
        rc = kc_hi / kc_lo
        khs = [kh_lo * (rh ** (i / STEPS)) for i in range(STEPS + 1)]
        kcs = [kc_lo * (rc ** (i / STEPS)) for i in range(STEPS + 1)]
        best_local = None
        for kh in khs:
            for kc in kcs:
                mism = 0
                for (S1, D1, S2, D2, w, Lo, Hi, label) in train:
                    _, pl = eval_job(S1, D1, S2, D2, w, Lo, Hi, kh, kc)
                    if pl != label:
                        mism += 1
                if best_local is None or mism < best_local[0]:
                    best_local = (mism, kh, kc)
        best = best_local
        _, kh_b, kc_b = best
        step_rh = rh ** (1.6 / STEPS)
        step_rc = rc ** (1.6 / STEPS)
        kh_lo, kh_hi = max(0.0005, kh_b / step_rh), kh_b * step_rh
        kc_lo, kc_hi = max(0.0001, kc_b / step_rc), kc_b * step_rc
    _, kh_fit, kc_fit = best

    out = []
    for (S1, D1, S2, D2, w, Lo, Hi) in query:
        nm, _ = eval_job(S1, D1, S2, D2, w, Lo, Hi, kh_fit, kc_fit)
        out.append("%.6f" % nm)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
