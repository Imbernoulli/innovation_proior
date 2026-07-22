# TIER: greedy
# The "obvious first attempt": fit ONE relaxation constant k from the training
# pass/fail tickets (a textbook single-parameter exponential-model fit via a
# coarse-to-fine LOG-SPACED grid search that minimizes label mismatches), but
# with two convenient simplifications that feel harmless: (a) heating and
# cooling share the same rate (no asymmetry), and (b) every hold is assumed to
# start from ambient T0 -- i.e. the ramp segment (S1,D1) is IGNORED even when
# present. Both simplifications are invisible on simple single-segment training
# tickets (which dominate any one class-balance snapshot) but compound badly on
# multi-segment schedules and on the short, wide-ranging held-out queries.
import math
import sys

T0 = 20.0


def naive_bounds(S2, D2, w, Lo, Hi, k):
    checkStart = D2 - w
    Ta = S2 + (T0 - S2) * math.exp(-k * checkStart)
    Tb = S2 + (T0 - S2) * math.exp(-k * D2)
    lo_t, hi_t = (Ta, Tb) if Ta <= Tb else (Tb, Ta)
    return lo_t, hi_t


def naive_label(S2, D2, w, Lo, Hi, k):
    lo_t, hi_t = naive_bounds(S2, D2, w, Lo, Hi, k)
    return 1 if (lo_t >= Lo and hi_t <= Hi) else 0


def naive_nm(S2, D2, w, Lo, Hi, k):
    lo_t, hi_t = naive_bounds(S2, D2, w, Lo, Hi, k)
    margin = min(lo_t - Lo, Hi - hi_t)
    return margin / ((Hi - Lo) / 2.0)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    testId = int(next(it)); N = int(next(it)); Q = int(next(it))
    train = []
    for _ in range(N):
        S1 = float(next(it)); D1 = float(next(it)); S2 = float(next(it)); D2 = float(next(it))
        w = float(next(it)); Lo = float(next(it)); Hi = float(next(it)); label = int(next(it))
        train.append((S2, D2, w, Lo, Hi, label))
    query = []
    for _ in range(Q):
        S1 = float(next(it)); D1 = float(next(it)); S2 = float(next(it)); D2 = float(next(it))
        w = float(next(it)); Lo = float(next(it)); Hi = float(next(it))
        query.append((S2, D2, w, Lo, Hi))

    klo, khi = 0.001, 2.0
    best = None
    STEPS = 30
    for _ in range(5):
        ratio = khi / klo
        cands = [klo * (ratio ** (i / STEPS)) for i in range(STEPS + 1)]
        best_local = None
        for k in cands:
            mism = 0
            for (S2, D2, w, Lo, Hi, label) in train:
                if naive_label(S2, D2, w, Lo, Hi, k) != label:
                    mism += 1
            if best_local is None or mism < best_local[0]:
                best_local = (mism, k)
        best = best_local
        step_ratio = ratio ** (1.6 / STEPS)
        klo, khi = max(0.0005, best[1] / step_ratio), best[1] * step_ratio
    k_fit = best[1]

    out = []
    for (S2, D2, w, Lo, Hi) in query:
        out.append("%.6f" % naive_nm(S2, D2, w, Lo, Hi, k_fit))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
