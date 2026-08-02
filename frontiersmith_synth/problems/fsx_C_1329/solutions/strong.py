# TIER: strong
"""The insight: fit and feasibility are a COUPLED choice, not a sequential
one. For every template, first clamp its kinetic sweet spot into the
*reachable* region (the intersection of the framework's crystallization
window with that template's own stability window -- an axis-aligned
rectangle, so the closest reachable point to the sweet spot is just an
independent per-axis clip). Then score the ACHIEVED yield (fit x reachable
proximity x removal-cost discount) at that point, and pick the template
that maximizes the true achievable yield across the whole library -- not
the template that maximizes raw geometric fit alone."""
import sys
import math

Q_NORM = 2.0
F_NORM = 1.0
T_NORM = 100.0
PH_NORM = 3.0


def f_ideal(c):
    return 0.15 * c + 0.1


def clamp(x, lo, hi):
    if hi < lo:
        return None
    return max(lo, min(hi, x))


def main():
    data = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = data[p]
        p += 1
        return v

    K = int(nxt())
    c = int(nxt())
    D_target = float(nxt())
    q_target = float(nxt())
    Tf_lo = float(nxt()); Tf_hi = float(nxt())
    pHf_lo = float(nxt()); pHf_hi = float(nxt())
    w1, w2, w3 = float(nxt()), float(nxt()), float(nxt())

    fid = f_ideal(c)

    best_yield = -1.0
    best_idx = -1
    best_pt = None

    for i in range(K):
        s = float(nxt()); q = float(nxt()); f = float(nxt())
        Tlo = float(nxt()); Thi = float(nxt())
        pHlo = float(nxt()); pHhi = float(nxt())
        Topt = float(nxt()); pHopt = float(nxt())
        R = float(nxt()); r = float(nxt())

        T_lo_int = max(Tf_lo, Tlo)
        T_hi_int = min(Tf_hi, Thi)
        pH_lo_int = max(pHf_lo, pHlo)
        pH_hi_int = min(pHf_hi, pHhi)
        if T_hi_int < T_lo_int or pH_hi_int < pH_lo_int:
            continue  # windows do not overlap: this template is unreachable

        T_star = clamp(Topt, T_lo_int, T_hi_int)
        pH_star = clamp(pHopt, pH_lo_int, pH_hi_int)

        size_m = max(0.0, 1.0 - abs(s - D_target) / D_target)
        charge_m = max(0.0, 1.0 - abs(q - q_target) / Q_NORM)
        shape_m = max(0.0, 1.0 - abs(f - fid) / F_NORM)
        sdi_val = w1 * size_m + w2 * charge_m + w3 * shape_m

        dT = (T_star - Topt) / T_NORM
        dpH = (pH_star - pHopt) / PH_NORM
        dist = math.sqrt(dT * dT + dpH * dpH)
        prox = max(0.0, 1.0 - dist / R)

        yield_val = sdi_val * prox * (1.0 - r)

        if yield_val > best_yield:
            best_yield = yield_val
            best_idx = i
            best_pt = (T_star, pH_star)

    print("%d %.6f %.6f" % (best_idx, best_pt[0], best_pt[1]))


if __name__ == "__main__":
    main()
