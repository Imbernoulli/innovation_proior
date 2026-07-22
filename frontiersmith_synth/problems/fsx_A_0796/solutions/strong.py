# TIER: strong
# Insight: the gear column s is not noise, it is the transducer's internal
# state -- each of the 3 gears applies its OWN linear-fractional map to n,
# so the rows must be split by s BEFORE the cross-multiply linearisation is
# applied, not after. Splitting first turns one hopelessly confounded system
# into three well-determined ones (each gear gets ~30 training rows for 3
# free unknowns), and a Mobius map recovered from finitely many EXACT points
# extrapolates correctly to any n because it is the same rational function,
# not a curve merely shaped to look right over the training range.
#
# The three per-gear curves are then combined into ONE algebraic expression
# in (n, s) using a degree-2 Lagrange selector on s (no branching needed,
# since s only ever takes the values 0, 1, 2):
#   L0(s) = (s-1)*(s-2)/2   L1(s) = s*(2-s)   L2(s) = s*(s-1)/2
# each 1 at its own gear index and 0 at the other two.
import sys


def solve(A, b):
    m = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(m):
        piv = max(range(c, m), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-15:
            d = 1e-15
        for r in range(m):
            if r == c:
                continue
            f = M[r][c] / d
            for k in range(c, m + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][m] / (M[i][i] if abs(M[i][i]) > 1e-15 else 1e-15) for i in range(m)]


def fit_gear(rows):
    # y = (n+B)/(C*n+D)  (numerator leading coeff fixed to 1; scale-invariant)
    m = 3
    Amat = [[0.0] * m for _ in range(m)]
    bb = [0.0] * m
    for n, y in rows:
        x = [1.0, -(n * y), -y]
        target = -n
        for r in range(m):
            bb[r] += x[r] * target
            for c in range(m):
                Amat[r][c] += x[r] * x[c]
    return solve(Amat, bb)


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n_rows = int(data[0])
    vals = data[2:]
    by_gear = {0: [], 1: [], 2: []}
    for i in range(n_rows):
        n = float(vals[3 * i])
        s = int(float(vals[3 * i + 1]))
        y = float(vals[3 * i + 2])
        by_gear[s].append((n, y))

    coefs = []
    for s in range(3):
        rows = by_gear.get(s, [])
        if len(rows) < 3:
            coefs.append((0.0, 1.0, 0.0))  # degenerate fallback
        else:
            coefs.append(tuple(fit_gear(rows)))

    (B0, C0, D0), (B1, C1, D1), (B2, C2, D2) = coefs
    expr = (
        "( ( (s-1) * (s-2) ) / 2 ) * ( (n+ %.10g ) / ( %.10g * n + %.10g ) )"
        " + ( s * (2-s) ) * ( (n+ %.10g ) / ( %.10g * n + %.10g ) )"
        " + ( ( s * (s-1) ) / 2 ) * ( (n+ %.10g ) / ( %.10g * n + %.10g ) )"
        % (B0, C0, D0, B1, C1, D1, B2, C2, D2)
    )
    print(expr)


if __name__ == "__main__":
    main()
