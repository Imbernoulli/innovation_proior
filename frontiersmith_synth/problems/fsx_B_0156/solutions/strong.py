# TIER: strong
# Recover the true saturating power law  eta = p + q*x**(-b):
# grid-search the exponent b, and for each b solve the linear least-squares
# for (p, q) on features [1, x**(-b)]. Pick the b with smallest train SSE.
import sys


def fit_linear(feat, ys):
    # feat: list of (1, f) rows -> solve p + q*f = y
    n = len(ys)
    sf = sum(f for _, f in feat)
    sff = sum(f * f for _, f in feat)
    sy = sum(ys)
    sfy = sum(feat[i][1] * ys[i] for i in range(n))
    det = n * sff - sf * sf
    if abs(det) < 1e-18:
        return None
    q = (n * sfy - sf * sy) / det
    p = (sy - q * sf) / n
    return p, q


def main():
    toks = sys.stdin.read().split()
    xs = [float(toks[i]) for i in range(0, len(toks), 2)]
    ys = [float(toks[i]) for i in range(1, len(toks), 2)]

    best = None
    b = 0.20
    while b <= 0.90 + 1e-9:
        feat = [(1.0, x ** (-b)) for x in xs]
        sol = fit_linear(feat, ys)
        if sol is not None:
            p, q = sol
            sse = 0.0
            for i, x in enumerate(xs):
                pred = p + q * (x ** (-b))
                d = pred - ys[i]
                sse += d * d
            if best is None or sse < best[0]:
                best = (sse, p, q, b)
        b += 0.01

    _, p, q, b = best
    print("%.10g + (%.10g)*x**(-%.10g)" % (p, q, b))


if __name__ == "__main__":
    main()
