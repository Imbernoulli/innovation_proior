# TIER: greedy
"""
Greedy / textbook approach: a single log-linear (pure exponential) fit.

log(A) = log(A0') + g'*t is linear in t. Ordinary least squares over ALL
training rows gives an excellent in-sample fit -- inside the training
window the true recursion's saturation term is a tiny fraction of the
step, so it looks like clean exponential growth. But this fit has NO
ceiling: it never reads t_B, s_hint or M_hint at all, so projected forward
it keeps compounding through the announced launch step and past the true
(lower, post-launch) ceiling.
"""
import sys, math


def ols(xs, ys):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        slope = 0.0
    else:
        slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def main():
    data = sys.stdin.read().split()
    idx = 0
    tid = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    tB = float(data[idx]); idx += 1
    s_hint = float(data[idx]); idx += 1
    M_hint = float(data[idx]); idx += 1
    ts, As = [], []
    for _ in range(n):
        ti = float(data[idx]); idx += 1
        A = float(data[idx]); idx += 1
        ts.append(ti); As.append(A)

    ys = [math.log(a) for a in As]
    g, log_a0 = ols(ts, ys)
    A0 = math.exp(log_a0)

    print("%.10g * expv(%.10g * t)" % (A0, g))


if __name__ == "__main__":
    main()
