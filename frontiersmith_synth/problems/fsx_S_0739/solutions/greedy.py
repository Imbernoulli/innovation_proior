# TIER: greedy
"""The obvious first move: notice the pooled data looks periodic-plus-drift,
so search for the single best-fitting frequency w and fit
y = A*sin(wx) + Q*cos(wx) + c*x + D directly against every regime's RAW rows
pooled together, via one global ordinary-least-squares fit (single D for
everybody). This gets the right IDEA (there is a shared oscillation) but
applies it naively: it never separates each regime's own (gain_r, offset_r),
so wildly different regimes' vertical shifts get crammed into one shared
offset D and often drag the frequency search toward the wrong w -- it just
throws every row into one bucket and searches for whatever curve looks best
across the jumble."""
import sys, math


def solve4(A, b):
    n = 4
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            M[piv][c] += 1e-9
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pv
        for r in range(n):
            if r != c:
                f = M[r][c]
                if f != 0.0:
                    for j in range(c, n + 1):
                        M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def global_fit(xs, ys, w):
    AtA = [[0.0] * 4 for _ in range(4)]
    Atb = [0.0] * 4
    for x, y in zip(xs, ys):
        f = [math.sin(w * x), math.cos(w * x), x, 1.0]
        for i in range(4):
            Atb[i] += f[i] * y
            for j in range(4):
                AtA[i][j] += f[i] * f[j]
    for i in range(4):
        AtA[i][i] += 1e-6
    A, Q, c, D = solve4(AtA, Atb)
    sse = 0.0
    for x, y in zip(xs, ys):
        pred = A * math.sin(w * x) + Q * math.cos(w * x) + c * x + D
        sse += (pred - y) ** 2
    return A, Q, c, D, sse


def main():
    data = sys.stdin.read().split()
    idx = 0
    K = int(data[idx]); idx += 1
    idx += 1  # test id
    xs, ys = [], []
    for _ in range(K):
        idx += 1  # regime id
        n = int(data[idx]); idx += 1
        idx += 1  # lo
        idx += 1  # hi
        for _ in range(n):
            x = float(data[idx]); idx += 1
            y = float(data[idx]); idx += 1
            xs.append(x)
            ys.append(y)

    best = None
    w = 0.75
    while w <= 1.95 + 1e-9:
        A, Q, c, D, sse = global_fit(xs, ys, w)
        if best is None or sse < best[0]:
            best = (sse, w, A, Q, c, D)
        w += 0.02
    _, w, A, Q, c, D = best

    print("%.6f*sin(%.6f*x) + %.6f*cos(%.6f*x) + %.6f*x + %.6f" % (A, w, Q, w, c, D))


if __name__ == "__main__":
    main()
