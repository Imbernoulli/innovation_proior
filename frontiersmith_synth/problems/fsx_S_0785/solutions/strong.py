# TIER: strong
# The insight: sampling the log only at phase 0 (tick = epoch*P) removes the
# fast inner stride entirely, leaving a[epoch*P] = B0 + B1*r + B2*r^2 +
# jitter(epoch*P) -- a clean quadratic in the epoch index r.  Fit that
# quadratic by least squares over the training phase-0 samples (many epochs
# -> jitter averages out and the integer coefficients round out exactly).
# Then recover the inner stride S0 from the residual interior deltas after
# subtracting the now-known drift.  Emit the closed-form law
#   B0 + B1*(t//P) + B2*(t//P)**2 + S0*(t%P)
# evaluated at the query tick -- this is causal and extrapolates correctly
# to epochs the training log never reached.  Slot 2 hedges +1 against the
# un-modelled (and, in this grammar, un-modellable) bit-mixed jitter term.
import sys


def fit_quadratic(xs, ys):
    n = len(xs)
    S0 = n
    S1 = sum(xs)
    S2 = sum(x * x for x in xs)
    S3 = sum(x ** 3 for x in xs)
    S4 = sum(x ** 4 for x in xs)
    T0 = sum(ys)
    T1 = sum(x * y for x, y in zip(xs, ys))
    T2 = sum(x * x * y for x, y in zip(xs, ys))
    A = [[S0, S1, S2, T0], [S1, S2, S3, T1], [S2, S3, S4, T2]]
    for i in range(3):
        piv = A[i][i]
        if abs(piv) < 1e-12:
            piv = 1e-12
        for j in range(i, 4):
            A[i][j] /= piv
        for k in range(3):
            if k != i:
                f = A[k][i]
                for j in range(i, 4):
                    A[k][j] -= f * A[i][j]
    return A[0][3], A[1][3], A[2][3]


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("SLOT1 h1")
        print("SLOT2 NONE")
        return
    N, P, tid = int(data[0]), int(data[1]), int(data[2])
    addrs = [int(x) for x in data[3:3 + N]]
    Rtrain = N // P

    xs = list(range(Rtrain))
    ys = [addrs[k * P] for k in range(Rtrain)]
    b0, b1, b2 = fit_quadratic(xs, ys)
    B0, B1, B2 = round(b0), round(b1), round(b2)

    diffs = []
    for t in range(N - 1):
        if (t + 1) % P == 0:
            continue
        r_t, r_t1 = t // P, (t + 1) // P
        trend_t = B0 + B1 * r_t + B2 * r_t * r_t
        trend_t1 = B0 + B1 * r_t1 + B2 * r_t1 * r_t1
        diffs.append((addrs[t + 1] - addrs[t]) - (trend_t1 - trend_t))
    S0 = round(sum(diffs) / len(diffs)) if diffs else 0

    law = "%d + %d * ( t // %d ) + %d * ( t // %d ) ** 2 + %d * ( t %% %d )" % (
        B0, B1, P, B2, P, S0, P)
    print("SLOT1 %s" % law)
    print("SLOT2 ( %s ) + 1" % law)


if __name__ == "__main__":
    main()
