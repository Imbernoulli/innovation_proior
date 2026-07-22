# TIER: strong
# INSIGHT: the statement GIVES us the exact scaling ansatz --
#   S(p) = A*(W*g((p-p_c)/W))^beta,  g(z) = 0.5*(z + sqrt(z*z+4))
# -- but a naive fit ignores it.  Taking the ansatz seriously and actually
# fitting its unknowns (p_c, beta, A; W is given) is the genuine insight: the
# sub-critical census is NOT a generic smooth curve, it is one asymptote of
# this scaling function, which reduces to (p-p_c)^beta well above p_c and
# decays smoothly below it.  Rather than fitting an unstructured flexible
# curve, GRID-SEARCH candidate p_c (must lie above the observed census) and
# beta, fit the amplitude in closed form (linear least squares given the
# candidate shape), keep the combination with lowest training residual, then
# locally refine.  This law carries the actual singularity, so unlike a smooth
# polynomial it extrapolates correctly through and beyond the (unknown)
# transition.
import sys, math


def read_rows():
    data = sys.stdin.read().split()
    n = int(data[1])
    w = float(data[2])
    rows = []
    for i in range(n):
        p = float(data[3 + 2 * i])
        s = float(data[3 + 2 * i + 1])
        rows.append((p, s))
    return rows, w


def gfun(z):
    return 0.5 * (z + math.sqrt(z * z + 4.0))


def fit_amp(rows, pcc, w, bc):
    num = 0.0
    den = 0.0
    for p, s in rows:
        z = (p - pcc) / w
        T = (w * gfun(z)) ** bc
        num += T * s
        den += T * T
    if den < 1e-18:
        return None
    a = num / den
    return a if a > 0 else None


def sse_on(rows, pcc, w, bc, a):
    sse = 0.0
    for p, s in rows:
        z = (p - pcc) / w
        T = (w * gfun(z)) ** bc
        sse += (a * T - s) ** 2
    return sse


def strong_fit(rows, w, p_max_train):
    pc_candidates = [p_max_train + d for d in
                      (0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
                       0.07, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30)]
    beta_candidates = [x / 100.0 for x in range(50, 225, 5)]  # 0.50 .. 2.20
    best = None
    for pcc in pc_candidates:
        for bc in beta_candidates:
            a = fit_amp(rows, pcc, w, bc)
            if a is None:
                continue
            z98 = (0.98 - pcc) / w
            shape98 = (w * gfun(z98)) ** bc
            if a * shape98 > 3.0:            # reject candidates that blow up
                continue
            sse = sse_on(rows, pcc, w, bc, a)
            if best is None or sse < best[0]:
                best = (sse, pcc, bc)
    if best is None:
        return p_max_train + 0.05, w, 1.0, 1.0
    _, pcc, bc = best
    for _ in range(4):
        improved = False
        for dpc in (-0.004, -0.001, 0.001, 0.004):
            npcc = pcc + dpc
            if npcc <= p_max_train:
                continue
            a = fit_amp(rows, npcc, w, bc)
            if a is None:
                continue
            sse = sse_on(rows, npcc, w, bc, a)
            if sse < best[0]:
                best = (sse, npcc, bc)
                pcc = npcc
                improved = True
        for db in (-0.03, -0.01, 0.01, 0.03):
            nbc = bc + db
            if nbc <= 0.2:
                continue
            a = fit_amp(rows, pcc, w, nbc)
            if a is None:
                continue
            sse = sse_on(rows, pcc, w, nbc, a)
            if sse < best[0]:
                best = (sse, pcc, nbc)
                bc = nbc
                improved = True
        if not improved:
            break
    amp = fit_amp(rows, pcc, w, bc)
    if amp is None:
        amp = 1.0
    return pcc, w, bc, amp


def main():
    rows, w = read_rows()
    p_max_train = max(p for p, s in rows)
    pcc, w, bc, amp = strong_fit(rows, w, p_max_train)
    # emit S(p) = amp * powv( w*0.5*( (p-pcc)/w + powv( ((p-pcc)/w)^2 + 4, 0.5 ) ), bc )
    # numeric constants are kept as their own whitespace-separated tokens (sign
    # folded in, not glued to parentheses) so each reads as a clean float literal.
    zterm = "( ( p - %.8f ) / %.8f )" % (pcc, w)
    expr = (
        "%.8f * powv( %.8f * ( %s + powv( %s * %s + 4.0 , 0.5 ) ) , %.6f )"
        % (amp, 0.5 * w, zterm, zterm, zterm, bc)
    )
    print(expr)


if __name__ == "__main__":
    main()
