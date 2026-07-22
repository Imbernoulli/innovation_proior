# TIER: greedy
"""The obvious first read of the log: treat the drift reading as a plain
linear function of the raw per-burst SYMBOL COUNTS (order-blind bag of
symbols), fit by ordinary least squares. This never models a persistent
polarity flag at all -- it has no notion that one code's effect on later
symbols can flip sign. Because logged training bursts are short (<=20
symbols) and rarely contain more than a couple of polarity pulses, this
fit tracks the noisy training rows reasonably well (an unconstrained
5-parameter OLS fit is, after all, explicitly minimizing training
residuals). Read the role assignment off the fitted coefficients: the
most positive coefficient becomes the PRESSURIZE code, the most negative
becomes VENT, and the remaining two are broken by |coefficient| into
POLARITY / NULL (larger magnitude -> POLARITY, on the belief that a
bigger leftover correlation means a bigger leftover effect). Always
assume the relay starts in normal polarity. This has no persistent-state
memory, so on long held-out bursts with many compounding polarity flips
-- especially the role-sorted 'clustered' ones -- its prediction stays
roughly proportional to raw symbol counts while the true drift depends
critically on flip ORDER, and it diverges badly."""
import sys

ALPHABET = "ABCD"
U_SUB_MIN, U_SUB_MAX = 1, 1000


def solve_linear(A, rhs):
    """Gaussian elimination with partial pivoting for a small dense system."""
    n = len(A)
    M = [row[:] + [rhs[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                if f != 0.0:
                    M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def main():
    header = sys.stdin.readline().split()
    K = int(header[1])
    rows = []
    for _ in range(K):
        parts = sys.stdin.readline().split()
        s, y = parts[0], int(parts[1])
        rows.append((s, y))

    # features: cA, cB, cC, cD, intercept
    feats = []
    ys = []
    for s, y in rows:
        counts = [s.count(ch) for ch in ALPHABET]
        feats.append(counts + [1.0])
        ys.append(float(y))

    n = 5
    AtA = [[0.0] * n for _ in range(n)]
    Aty = [0.0] * n
    for f, y in zip(feats, ys):
        for i in range(n):
            Aty[i] += f[i] * y
            for j in range(n):
                AtA[i][j] += f[i] * f[j]
    # tiny ridge regularization for numerical stability
    for i in range(n):
        AtA[i][i] += 1e-6

    beta = solve_linear(AtA, Aty)
    coeffs = dict(zip(ALPHABET, beta[:4]))

    order = sorted(ALPHABET, key=lambda ch: coeffs[ch])
    dec_sym = order[0]
    inc_sym = order[3]
    rest = [order[1], order[2]]
    rest.sort(key=lambda ch: -abs(coeffs[ch]))
    flip_sym, noop_sym = rest[0], rest[1]

    u_est = (coeffs[inc_sym] - coeffs[dec_sym]) / 2.0
    u = max(U_SUB_MIN, min(U_SUB_MAX, int(round(u_est))))
    if u < 1:
        u = 1
    p0 = 1

    print(inc_sym, dec_sym, flip_sym, noop_sym, u, p0)


if __name__ == "__main__":
    main()
