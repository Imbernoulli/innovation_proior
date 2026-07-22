# TIER: greedy
# The obvious first model for a census: an ORDINARY (undelayed) seasonal logistic
#   x[t+1] = r[s] * x[t] * (1 - x[t]/K)   ==   a_s*x + g_s*(x*x)   per season s.
# Fit the four seasons by least squares.  It tracks the quiescent training census
# to the noise floor -- so a practitioner stops here.  But an undelayed law can
# only relax smoothly; on the held-out CRASH the true system rings and overshoots,
# which this law cannot reproduce.  It never suspects a delay.
import sys

S = 4


def read_census():
    data = sys.stdin.read().split()
    t, n, s, maxlag = int(data[0]), int(data[1]), int(data[2]), int(data[3])
    xs = [float(v) for v in data[4:4 + n]]
    return xs


def fit_undelayed(xs):
    # per season: y = a*f1 + g*f2, f1=x, f2=x*x  (least squares, 2 params)
    A = {s: [[0.0, 0.0], [0.0, 0.0]] for s in range(S)}
    b = {s: [0.0, 0.0] for s in range(S)}
    for i in range(len(xs) - 1):
        s = i % S
        xi = xs[i]
        f1, f2, y = xi, xi * xi, xs[i + 1]
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


def main():
    xs = read_census()
    coef = fit_undelayed(xs)
    terms = []
    for s in range(S):
        a, g = coef[s]
        terms.append("c%d * ( %.8f * x + %.8f * x * x )" % (s, a, g))
    print(" + ".join(terms))


if __name__ == "__main__":
    main()
