import sys, math

TOL = 1e-6


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def water_reduction(p, wr_max, p_half):
    if p <= 0.0:
        return 0.0
    return wr_max * p / (p + p_half)


def scr_of(w, c, p, rho_c, rho_w, air, k1, k2, k3, k4):
    vc = c / rho_c
    vw = w / rho_w
    vagg = 1.0 - air - vc - vw
    val = k1 * (w / c) + k2 * (vc + vw) - k3 * vagg + k4 * p
    return val, vagg


def main():
    inp_tokens = open(sys.argv[1]).read().split()
    try:
        it = iter(inp_tokens)
        K = int(next(it))
        rho_c = float(next(it)); rho_w = float(next(it)); air = float(next(it))
        c_min = float(next(it)); c_max = float(next(it))
        w_min = float(next(it)); w_max = float(next(it))
        wc_min = float(next(it)); wc_max = float(next(it))
        wr_max = float(next(it)); p_half = float(next(it)); p_max = float(next(it))
        k1 = float(next(it)); k2 = float(next(it)); k3 = float(next(it)); k4 = float(next(it))
        A = float(next(it)); B = float(next(it))
        vagg_min = float(next(it)); risk_limit = float(next(it))
        W0 = [float(next(it)) for _ in range(K)]
    except Exception:
        fail("bad input")

    # ---- internal baseline B_check: reference blend (j=1), no admixture, minimum
    # cement, water = exactly the amount that blend needs for workability. This recipe
    # is always feasible by construction (see gen.py's calibration). ----
    w0_1 = W0[0]
    base_wc = w0_1 / c_min
    base_scr, base_vagg = scr_of(w0_1, c_min, 0.0, rho_c, rho_w, air, k1, k2, k3, k4)
    base_ok = (c_min - TOL <= c_min <= c_max + TOL and w_min - TOL <= w0_1 <= w_max + TOL
               and wc_min - TOL <= base_wc <= wc_max + TOL
               and base_vagg >= vagg_min - TOL and base_scr <= risk_limit + TOL)
    if not base_ok:
        fail("internal baseline infeasible (problem miscalibrated)")
    F_base = A - B * base_wc
    B_check = max(1e-9, F_base)

    # ---- parse participant output: "j c w p" ----
    out_tokens = open(sys.argv[2]).read().split()
    if len(out_tokens) != 4:
        fail("expected exactly 4 numbers: j c w p, got %d" % len(out_tokens))
    try:
        j_raw = out_tokens[0]
        j = int(j_raw)
        c = float(out_tokens[1])
        w = float(out_tokens[2])
        p = float(out_tokens[3])
    except Exception:
        fail("parse error")

    for name, v in (("c", c), ("w", w), ("p", p)):
        if not math.isfinite(v):
            fail("non-finite %s" % name)
    if not (1 <= j <= K):
        fail("blend index j=%d out of range [1,%d]" % (j, K))

    if not (c_min - TOL <= c <= c_max + TOL):
        fail("cement content c=%.6f out of [%.6f,%.6f]" % (c, c_min, c_max))
    if not (w_min - TOL <= w <= w_max + TOL):
        fail("water content w=%.6f out of [%.6f,%.6f]" % (w, w_min, w_max))
    if not (-TOL <= p <= p_max + TOL):
        fail("admixture dosage p=%.6f out of [0,%.6f]" % (p, p_max))
    if c <= 0:
        fail("non-positive cement content")

    wc = w / c
    if not (wc_min - TOL <= wc <= wc_max + TOL):
        fail("water/cement ratio %.6f out of [%.6f,%.6f]" % (wc, wc_min, wc_max))

    # workability: chosen water must cover the (admixture-reduced) demand of the
    # chosen aggregate blend
    W0_j = W0[j - 1]
    req = W0_j * (1.0 - water_reduction(p, wr_max, p_half))
    if w < req - TOL:
        fail("workability failed: w=%.6f < required %.6f for blend %d at dosage p=%.6f"
             % (w, req, j, p))

    scr_val, vagg = scr_of(w, c, p, rho_c, rho_w, air, k1, k2, k3, k4)
    if vagg < vagg_min - TOL:
        fail("aggregate volume fraction %.6f below floor %.6f (paste-flooded mix)" % (vagg, vagg_min))
    if scr_val > risk_limit + TOL:
        fail("shrinkage-cracking risk %.6f exceeds budget %.6f" % (scr_val, risk_limit))

    F = A - B * wc
    if not math.isfinite(F) or F <= 0:
        fail("non-positive/non-finite strength")

    sc = min(1000.0, 100.0 * F / B_check)
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B_check, sc / 1000.0))


if __name__ == "__main__":
    main()
