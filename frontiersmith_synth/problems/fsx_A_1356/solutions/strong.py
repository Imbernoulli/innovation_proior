# TIER: strong
"""
The genuine insight: a strategy's quality is governed only by its worst-case
column, so the right move is not to model the opponent's behavior at all but
to solve the maximin LP directly --

    maximize t   s.t.   sum_i p_i * A[i][j] >= t  for every column j
                         sum_i p_i = 1,  p_i >= 0

An LP solve identifies the EQUILIBRIUM SUPPORT (which rows get positive
weight, exposed by which column constraints bind at the optimum) instead of
chasing whichever rows look attractive against sampled/historical opponent
behavior. Rows that are only good against the observed log but hide a weak
column get correctly assigned zero weight, because they cannot help the
binding (tight) columns.

Falls back to a multiplicative-weights (no-regret) self-play solver, which
also converges to the minimax value, if scipy is unavailable.
"""
import sys


def solve_lp(m, n, A):
    from scipy.optimize import linprog

    c = [0.0] * m + [-1.0]
    A_ub = []
    b_ub = []
    for j in range(n):
        row = [-float(A[i][j]) for i in range(m)] + [1.0]
        A_ub.append(row)
        b_ub.append(0.0)
    A_eq = [[1.0] * m + [0.0]]
    b_eq = [1.0]
    bounds = [(0, None)] * m + [(0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("linprog failed")
    p = list(res.x[:m])
    s = sum(max(0.0, x) for x in p)
    if s <= 0:
        raise RuntimeError("degenerate LP solution")
    return [max(0.0, x) / s for x in p]


def solve_mwu(m, n, A):
    """No-regret self-play fallback: multiplicative-weights updates for both
    players converge (time-averaged) to the minimax equilibrium value."""
    T = 4000
    row_w = [1.0] * m
    col_w = [1.0] * n
    row_cum = [0.0] * m
    amax = max(max(r) for r in A)
    eta = (2.0 * (0.5 ** 0.5)) / amax  # ~ sqrt(2 ln N / T)-scale, fixed small step

    for _ in range(T):
        rs = sum(row_w)
        cs = sum(col_w)
        rp = [x / rs for x in row_w]
        cp = [x / cs for x in col_w]
        for i in range(m):
            row_cum[i] += rp[i]
        # row player: multiplicatively favor rows that scored well vs current cp
        row_gain = [sum(A[i][j] * cp[j] for j in range(n)) for i in range(m)]
        gmax = max(row_gain) or 1.0
        for i in range(m):
            row_w[i] *= (1.0 + eta * (row_gain[i] / gmax))
        # column player: multiplicatively favor columns that hurt row player most
        col_loss = [sum(A[i][j] * rp[i] for i in range(m)) for j in range(n)]
        lmax = max(col_loss) or 1.0
        for j in range(n):
            col_w[j] *= (1.0 + eta * (col_loss[j] / lmax))
    s = sum(row_cum)
    return [x / s for x in row_cum]


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    m = int(nxt())
    n = int(nxt())
    A = [[int(nxt()) for _ in range(n)] for _ in range(m)]
    # history is deliberately unused: it describes a past opponent, not the
    # worst-case adversary this strategy must be robust against.

    try:
        p = solve_lp(m, n, A)
    except Exception:
        p = solve_mwu(m, n, A)

    print(" ".join(f"{x:.9f}" for x in p))


if __name__ == "__main__":
    main()
