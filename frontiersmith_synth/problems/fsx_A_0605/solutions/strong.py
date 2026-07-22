# TIER: strong
# INSIGHT: the quiescent census is fit to the noise floor by an undelayed
# logistic, so fit QUALITY alone will not betray the mechanism.  The tell of the
# hidden delay lives in the RESIDUAL STRUCTURE: the crowding term acts on the
# density some tau seasons ago, so among all candidate lags, the delayed-crowding
# regressor x[t]*x[t-d] that best explains the one-step residual of the census is
# the true delay.  We probe every lag d=1..MAXLAG by refitting the seasonal law
#   x[t+1] = a_s*x[t] + g_s*(x[t]*x[t-d])
# and keep the delay whose law leaves the least unexplained variation; then emit
# that delayed seasonal law.  Unlike the undelayed fit, it carries the delay that
# makes the held-out crash RING -- so it extrapolates into the oscillatory regime.
import sys

S = 4


def read_census():
    data = sys.stdin.read().split()
    n, maxlag = int(data[1]), int(data[3])
    xs = [float(v) for v in data[4:4 + n]]
    return xs, maxlag


def fit_delay(xs, d):
    A = {s: [[0.0, 0.0], [0.0, 0.0]] for s in range(S)}
    b = {s: [0.0, 0.0] for s in range(S)}
    for i in range(len(xs) - 1):
        if d > 0 and i - d < 0:
            continue
        s = i % S
        xi = xs[i]
        xlag = xs[i] if d == 0 else xs[i - d]
        f1, f2, y = xi, xi * xlag, xs[i + 1]
        A[s][0][0] += f1 * f1; A[s][0][1] += f1 * f2
        A[s][1][0] += f2 * f1; A[s][1][1] += f2 * f2
        b[s][0] += f1 * y;     b[s][1] += f2 * y
    coef = {}
    for s in range(S):
        m, rhs = A[s], b[s]
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        if abs(det) < 1e-12:
            coef[s] = (1.0, 0.0)
        else:
            a = (rhs[0] * m[1][1] - rhs[1] * m[0][1]) / det
            g = (m[0][0] * rhs[1] - m[1][0] * rhs[0]) / det
            coef[s] = (a, g)
    return coef


def resid_var(xs, d, coef):
    sse, cnt = 0.0, 0
    for i in range(len(xs) - 1):
        if d > 0 and i - d < 0:
            continue
        s = i % S
        xi = xs[i]
        xlag = xs[i] if d == 0 else xs[i - d]
        a, g = coef[s]
        sse += (xs[i + 1] - (a * xi + g * xi * xlag)) ** 2
        cnt += 1
    return sse / max(1, cnt)


def main():
    xs, maxlag = read_census()
    best_d, best_v, best_coef = 1, None, None
    for d in range(1, maxlag + 1):
        coef = fit_delay(xs, d)
        v = resid_var(xs, d, coef)
        if best_v is None or v < best_v:
            best_v, best_d, best_coef = v, d, coef
    tau = best_d
    terms = []
    for s in range(S):
        a, g = best_coef[s]
        terms.append("c%d * ( %.8f * x + %.8f * x * lag%d )" % (s, a, g, tau))
    print(" + ".join(terms))


if __name__ == "__main__":
    main()
