# TIER: greedy
# A LOGISTIC fit whose plateau K is OPTIMISED (grid-searched) on the early data instead of
# fixed at 1.05*max like the baseline.  A better plateau estimate improves the late-time
# prediction and beats the trivial baseline -- but the logistic S-curve is SYMMETRIC, whereas
# the true growth is an asymmetric Gompertz curve, so its tail shape stays wrong and it loses
# to the correct-family strong fit.
import sys, math


def lstsq(rows, y):
    m = len(rows[0])
    A = [[0.0] * m for _ in range(m)]
    bvec = [0.0] * m
    for r, yy in zip(rows, y):
        for i in range(m):
            bvec[i] += r[i] * yy
            for j in range(m):
                A[i][j] += r[i] * r[j]
    M = [A[i][:] + [bvec[i]] for i in range(m)]
    for c in range(m):
        piv = max(range(c, m), key=lambda rr: abs(M[rr][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(m):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, m + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][m] / M[i][i] for i in range(m)]


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    idx = 1
    pts = []
    for _ in range(n):
        t = float(toks[idx]); nn = float(toks[idx + 1])
        idx += 2
        pts.append((t, nn))
    nmax = max(nn for _, nn in pts)

    best = None
    steps = 80
    for i in range(steps + 1):
        K = nmax * (1.02 + (3.5 - 1.02) * i / steps)   # plateau grid: 1.02x .. 3.5x max
        rows, z = [], []
        ok = True
        for (t, nn) in pts:
            p = nn / K
            if p <= 0.0 or p >= 1.0:
                ok = False
                break
            rows.append([1.0, t])
            z.append(math.log(p / (1.0 - p)))
        if not ok:
            continue
        co = lstsq(rows, z)
        if co is None:
            continue
        a0, a1 = co
        err = 0.0
        for (t, nn) in pts:
            s = min(max(a0 + a1 * t, -60.0), 60.0)
            pred = K / (1.0 + math.exp(-s))
            err += (pred - nn) ** 2
        if best is None or err < best[0]:
            best = (err, K, a0, a1)

    _, K, a0, a1 = best
    print("%.10g / (1 + exp(-(%.10g + %.10g * t)))" % (K, a0, a1))


if __name__ == "__main__":
    main()
