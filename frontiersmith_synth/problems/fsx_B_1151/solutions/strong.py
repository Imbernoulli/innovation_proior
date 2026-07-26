# TIER: strong
# The insight: don't fit T as a function of (p1,p2,d) at all -- IDENTIFY the
# hidden dynamical system instead. Every training row hands us a full noisy
# proportion trace, and consecutive TRIPLES (p_{k-1},p_k,p_{k+1}) obey a
# STATIONARY linear recurrence p_{k+1} = alpha*p_k + beta*p_{k-1} + gamma
# shared by every design in this test case. Pool ALL such triples across ALL
# rows (even depth-3..6 rows give some) into one big least-squares system for
# (alpha,beta,gamma) -- shallow depth is already enough to pin the spectrum
# down, because the recurrence only needs 3 consecutive points, not many
# depths. From alpha,beta recover the characteristic roots lambda1,lambda2
# (the spectrum) and the fixed point p* = gamma/(1-alpha-beta). Then the
# general solution p_k = p* + A*lambda1**k + B*lambda2**k gives the deepest
# frame's deviation in CLOSED FORM: p_d-p* = A*lambda1**d + B*lambda2**d, so
# T(d) = (p_d-p*)^2 expands into A^2*lambda1**(2d) + 2AB*(lambda1*lambda2)**d
# + B^2*lambda2**(2d) -- exactly the exponential envelope a response surface
# that is merely polynomial/log-linear in d cannot represent, and A,B are
# linear in p1,p2 (via Cramer's rule on d1=p1-p*, d2=p2-p*) so each design's
# OWN amplitude is captured too. The final expression is a single arithmetic
# formula in p1,p2,d with the recovered constants baked in.
import sys, re


def spacify(expr):
    toks = re.findall(r"\*\*|\d+\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?"
                       r"|[A-Za-z_]\w*|[+\-*/(),]", expr)
    return " ".join(toks)


def solve_normal_eq(X, y):
    m = len(X[0])
    XtX = [[0.0] * m for _ in range(m)]
    Xty = [0.0] * m
    for row, yv in zip(X, y):
        for i in range(m):
            Xty[i] += row[i] * yv
            for j in range(m):
                XtX[i][j] += row[i] * row[j]
    for i in range(m):
        XtX[i][i] += 1e-9
    A = [XtX[i][:] + [Xty[i]] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        if abs(pv) < 1e-12:
            pv = 1e-12
        for j in range(col, m + 1):
            A[col][j] /= pv
        for r in range(m):
            if r != col:
                f = A[r][col]
                if f != 0.0:
                    for j in range(col, m + 1):
                        A[r][j] -= f * A[col][j]
    return [A[i][m] for i in range(m)]


def guard_denom(v, eps=1e-6):
    if abs(v) < eps:
        return eps if v >= 0 else -eps
    return v


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.0")
        return
    n = int(data[0])
    vals = data[2:]
    rows = []
    idx = 0
    for _ in range(n):
        dep = int(vals[idx])
        row = vals[idx: idx + dep + 2]
        seq = [float(x) for x in row[1:1 + dep]]
        rows.append(seq)
        idx += dep + 2

    # pool consecutive triples across all rows: p_{k+1} = alpha*p_k+beta*p_{k-1}+gamma
    X, y = [], []
    for seq in rows:
        for k in range(1, len(seq) - 1):
            X.append([seq[k], seq[k - 1], 1.0])
            y.append(seq[k + 1])

    if len(X) < 3:
        fallback = sum(seq[-1] for seq in rows) / len(rows) if rows else 0.0
        print(spacify("(%.10g)" % fallback))
        return

    alpha, beta, gamma = solve_normal_eq(X, y)

    disc = alpha * alpha + 4.0 * beta
    if disc < 1e-6:
        disc = 1e-6
    sq = disc ** 0.5
    r1 = (alpha + sq) / 2.0
    r2 = (alpha - sq) / 2.0
    if abs(r1) >= abs(r2):
        lam1, lam2 = r1, r2
    else:
        lam1, lam2 = r2, r1
    if abs(lam1 - lam2) < 1e-4:
        lam2 = lam1 - 1e-4

    denom_fix = guard_denom(1.0 - alpha - beta)
    pstar = gamma / denom_fix
    # keep it in a physically sane range (proportions observed in [0.05,0.98])
    pstar = min(0.98, max(0.05, pstar))

    det = guard_denom(lam1 * lam2 * lam2 - lam2 * lam1 * lam1)
    c11 = (lam2 * lam2) / det
    c12 = -lam2 / det
    c21 = -(lam1 * lam1) / det
    c22 = lam1 / det

    A_expr = "((%.10g)*(p1-(%.10g))+(%.10g)*(p2-(%.10g)))" % (c11, pstar, c12, pstar)
    B_expr = "((%.10g)*(p1-(%.10g))+(%.10g)*(p2-(%.10g)))" % (c21, pstar, c22, pstar)
    T11 = "((%.10g)**(2*d))" % lam1
    T22 = "((%.10g)**(2*d))" % lam2
    T12 = "(((%.10g)*(%.10g))**d)" % (lam1, lam2)

    expr = "(%s)**2*%s + 2*%s*%s*%s + (%s)**2*%s" % (
        A_expr, T11, A_expr, B_expr, T12, B_expr, T22)
    print(spacify(expr))


if __name__ == "__main__":
    main()
