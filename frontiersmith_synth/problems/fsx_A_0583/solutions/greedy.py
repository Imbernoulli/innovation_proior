# TIER: greedy
# The obvious recipe: assume the ledger obeys ONE single linear recurrence and
# recover it by fitting the longest recurrence that explains the window --
#     a(n) = c1*a(n-1) + c2*a(n-2) + ... + cp*a(n-p) + c0
# choosing the order p with the best least-squares training fit, then rolling it
# forward K steps.  Because interleaving a few short laws IS a longer linear law,
# this fits the finite window well and even nails the first extrapolated steps --
# but it is blind to the residue-class structure and, being ordinary least
# squares, is dragged off by the few far-off corrupted lines, so the recovered
# coefficients are slightly wrong and the roll-forward derails within a handful
# of steps.  A single long recurrence is exactly the mask the two short laws wear.
import sys


def solve_ls(M, y, d):
    A = [row[:] + [y[i]] for i, row in enumerate(M)]
    for c in range(d):
        piv = max(range(c, d), key=lambda r: abs(A[r][c]))
        if abs(A[piv][c]) < 1e-9:
            A[c][c] += 1e-6                    # ridge nudge on singular pivot
        A[c], A[piv] = A[piv], A[c]
        for r in range(d):
            if r != c:
                f = A[r][c] / A[c][c]
                for k in range(c, d + 1):
                    A[r][k] -= f * A[c][k]
    return [A[i][d] / A[i][i] for i in range(d)]


def fit_order(a, N, p):
    d = p + 1
    M = [[0.0] * d for _ in range(d)]
    yv = [0.0] * d
    for n in range(p + 1, N + 1):
        feat = [a[n - 1 - k] for k in range(p)] + [1.0]
        for i in range(d):
            for j in range(d):
                M[i][j] += feat[i] * feat[j]
            yv[i] += feat[i] * a[n]
    coef = solve_ls(M, yv, d)
    # training residual
    err = 0.0
    for n in range(p + 1, N + 1):
        feat = [a[n - 1 - k] for k in range(p)] + [1.0]
        pr = sum(coef[i] * feat[i] for i in range(d))
        err += (pr - a[n]) ** 2
    return coef, err / max(1, N - p)


def main():
    data = sys.stdin.read().split()
    N, K = int(data[0]), int(data[1])
    a = [0.0] + [float(x) for x in data[3:3 + N]]

    best = None
    for p in range(2, 7):
        coef, err = fit_order(a, N, p)
        # light complexity tax so it doesn't just pick the largest order
        score = err * (1.0 + 0.02 * p)
        if best is None or score < best[0]:
            best = (score, p, coef)
    _, p, coef = best

    hist = [a[N - k] for k in range(p)]           # a[N], a[N-1], ... newest first
    preds = []
    for _ in range(K):
        v = sum(coef[i] * hist[i] for i in range(p)) + coef[p]
        if v != v or abs(v) > 1e18:
            v = hist[0]
        preds.append(int(round(v)))
        hist = [v] + hist[:-1]
    sys.stdout.write("\n".join(str(x) for x in preds) + "\n")


if __name__ == "__main__":
    main()
