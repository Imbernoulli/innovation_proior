# TIER: strong
# The insight: don't fit a power law in the four raw knobs -- first do the
# dimensional analysis. The grading matrix U (3x4 integers) is given in the
# input; any response that is grading-neutral must, in log space, have its
# knob-exponents lying in the null space of U. That null space is EXACTLY
# 1-dimensional here (rank(U) = 3 by construction), so there is a UNIQUE (up
# to overall scale) integer exponent vector b with U @ b = 0 -- found by
# exact rational Gaussian elimination, no data needed at all for the
# DIRECTION. Only the single scalar exponent p (and the amplitude C) remain
# free, fit by an ordinary 1-D least-squares regression of log y against
# log(Pi), Pi = prod x_i**b_i. Because the direction itself is derived
# exactly rather than estimated, this single free parameter is immune to the
# orthogonal-noise contamination that sinks the raw four-parameter fit, and
# the same closed form extrapolates correctly wherever the held-out grid
# pushes the four knobs.
import sys, math
from fractions import Fraction
from math import gcd


def nullspace_vector(U):
    """Exact rational nullspace of a full-row-rank r x n matrix with
    nullity 1 (guaranteed here: 3 rows, rank 3, 4 columns)."""
    A = [[Fraction(v) for v in row] for row in U]
    rows, cols = len(A), len(A[0])
    pivot_cols = []
    r = 0
    for c in range(cols):
        piv = None
        for rr in range(r, rows):
            if A[rr][c] != 0:
                piv = rr
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][c]
        A[r] = [v / pv for v in A[r]]
        for rr in range(rows):
            if rr != r and A[rr][c] != 0:
                f = A[rr][c]
                A[rr] = [A[rr][k] - f * A[r][k] for k in range(cols)]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    free_cols = [c for c in range(cols) if c not in pivot_cols]
    fc = free_cols[0]
    vec = [Fraction(0)] * cols
    vec[fc] = Fraction(1)
    for i, pc in enumerate(pivot_cols):
        vec[pc] = -A[i][fc]
    lcm = 1
    for f in vec:
        d = f.denominator
        lcm = lcm * d // gcd(lcm, d)
    ivec = [int(f * lcm) for f in vec]
    g = 0
    for v in ivec:
        g = gcd(g, abs(v))
    if g > 1:
        ivec = [v // g for v in ivec]
    return ivec


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    U = [[int(v) for v in data[2 + 4 * j: 2 + 4 * j + 4]] for j in range(3)]
    vals = data[2 + 12:]
    rows = []
    for i in range(n):
        x = [float(vals[5 * i + k]) for k in range(4)]
        y = float(vals[5 * i + 4])
        rows.append((x, y))

    b = nullspace_vector(U)

    logpi = []
    logy = []
    for x, y in rows:
        Pi = 1.0
        for xi, bi in zip(x, b):
            Pi *= xi ** bi
        logpi.append(math.log(Pi))
        logy.append(math.log(y))

    mx = sum(logpi) / n
    my = sum(logy) / n
    cov = sum((lp - mx) * (ly - my) for lp, ly in zip(logpi, logy))
    varx = sum((lp - mx) ** 2 for lp in logpi)
    if abs(varx) < 1e-18:
        varx = 1e-18
    p_hat = cov / varx
    C_hat = math.exp(my - p_hat * mx)

    print("%.10g * (x1**(%d) * x2**(%d) * x3**(%d) * x4**(%d))**(%.10g)"
          % (C_hat, b[0], b[1], b[2], b[3], p_hat))


if __name__ == "__main__":
    main()
