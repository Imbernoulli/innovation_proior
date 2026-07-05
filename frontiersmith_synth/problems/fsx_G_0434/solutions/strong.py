# TIER: strong
# Two-compartment (biexponential) recovery: C(t) = A*exp(-alpha t) + B*exp(-beta t).
# Grid-search the two rates (alpha,beta); for each rate pair the amplitudes
# (A,B) are the closed-form nonnegative least-squares solution in linear space,
# and the pair is scored by TRAIN log-MSE (the metric that weights the slow
# tail).  This separates the terminal elimination rate beta from the early
# mixed signal, so the law extrapolates into the late clearance tail.
import sys, math


def solve2(basis1, basis2, ys):
    # normal equations for [A,B] minimising sum (A*b1+B*b2 - y)^2
    s11 = sum(b * b for b in basis1)
    s22 = sum(b * b for b in basis2)
    s12 = sum(basis1[i] * basis2[i] for i in range(len(ys)))
    r1 = sum(basis1[i] * ys[i] for i in range(len(ys)))
    r2 = sum(basis2[i] * ys[i] for i in range(len(ys)))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-15:
        return None
    A = (r1 * s22 - r2 * s12) / det
    B = (s11 * r2 - s12 * r1) / det
    return A, B


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = data[2:]
    ts, cs = [], []
    for i in range(n):
        ts.append(float(vals[2 * i]))
        cs.append(float(vals[2 * i + 1]))
    logs = [math.log(max(c, 1e-9)) for c in cs]

    best = None
    # alpha (fast) in [0.6,3.4], beta (slow) in [0.05,0.6], alpha>beta
    al_grid = [0.6 + 0.05 * i for i in range(57)]      # up to ~3.4
    be_grid = [0.05 + 0.01 * i for i in range(56)]     # up to ~0.60
    for al in al_grid:
        b1 = [math.exp(-al * tt) for tt in ts]
        for be in be_grid:
            if be >= al - 0.15:
                continue
            b2 = [math.exp(-be * tt) for tt in ts]
            sol = solve2(b1, b2, cs)
            if sol is None:
                continue
            A, B = sol
            if A <= 0 or B <= 0:
                continue
            se = 0.0
            for i in range(n):
                pred = A * b1[i] + B * b2[i]
                if pred <= 0:
                    se = float("inf")
                    break
                d = math.log(pred) - logs[i]
                se += d * d
            if best is None or se < best[0]:
                best = (se, A, al, B, be)

    if best is None:
        # fallback: mono-exponential
        mt = sum(ts) / n
        my = sum(logs) / n
        sxx = sum((x - mt) ** 2 for x in ts)
        sxy = sum((ts[i] - mt) * (logs[i] - my) for i in range(n))
        c1 = sxy / sxx if sxx > 0 else 0.0
        c0 = my - c1 * mt
        print("0.0 + %r*exp(%r*t)" % (math.exp(c0), c1))
        return

    _, A, al, B, be = best
    print("0.0 + %r*exp(%r*t) + %r*exp(%r*t)" % (A, -al, B, -be))


if __name__ == "__main__":
    main()
