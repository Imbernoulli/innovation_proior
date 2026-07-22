# TIER: greedy
# The obvious recipe: this looks like a period-finding problem, so run a
# textbook periodogram -- grid-search a SINGLE stationary sinusoidal period
# (searched over the WHOLE plausible range, including periods as long as the
# entire training span) and add it to the linear ephemeris. This fits the
# training rows very well in every case, because when the true wobble period
# happens to be shorter than the campaign it is found correctly, and when it
# is LONGER than the campaign (so less than one cycle is ever observed) the
# search just latches onto some huge candidate period whose slow local curve
# soaks up the actual secular drift just as well, in-window. The blind spot:
# a fitted sinusoid, however long its period, stays BOUNDED forever -- it
# never reproduces a timing offset that keeps GROWING out at the far-future
# grading cycles the way genuine secular drift does. There is no notion of an
# explicit, ever-growing term anywhere in this recipe.
import sys, math


def lstsq(rows, y):
    m = len(rows[0])
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, yy in zip(rows, y):
        for r in range(m):
            b[r] += x[r] * yy
            for c in range(m):
                A[r][c] += x[r] * x[c]
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    n = m
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-14:
            d = 1e-14
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / d
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] if abs(M[i][i]) > 1e-14 else 1e-14) for i in range(n)]


def sse_periodic(obs, M):
    rows = [[1.0, k, math.sin(2 * math.pi * k / M), math.cos(2 * math.pi * k / M)]
             for k, _ in obs]
    ys = [v for _, v in obs]
    coef = lstsq(rows, ys)
    sse = sum((sum(c * x for c, x in zip(coef, row)) - y) ** 2 for row, y in zip(rows, ys))
    return sse, coef


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n = int(header[0])
    obs = []
    for ln in data[1:1 + n]:
        parts = ln.split()
        if len(parts) == 2:
            obs.append((int(parts[0]), float(parts[1])))
    if not obs:
        print("0.0")
        return
    Ktrainmax = max(k for k, _ in obs)

    Mgrid_lo = 3.0
    Mgrid_hi = max(6.0, float(Ktrainmax))
    ngrid = 140
    best = None
    for i in range(ngrid):
        Mc = Mgrid_lo + (Mgrid_hi - Mgrid_lo) * i / (ngrid - 1)
        sse, coef = sse_periodic(obs, Mc)
        if best is None or sse < best[0]:
            best = (sse, Mc, coef)
    _, Mbest, (a, b, As, Ac) = best
    print("%.10g + %.10g*k + %.10g*sin(2*pi*k/%.10g) + %.10g*cos(2*pi*k/%.10g)"
          % (a, b, As, Mbest, Ac, Mbest))


if __name__ == "__main__":
    main()
