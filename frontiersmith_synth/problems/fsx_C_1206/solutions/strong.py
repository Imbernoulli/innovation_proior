# TIER: strong
# The insight: the small-N miss-rate log cannot see the cliff (every row is
# deep in the sub-cliff tail, swamped by finite-sample noise), but the
# reuse-distance HISTOGRAM can -- its shape is a scale-free property of the
# workload's locality, measurable regardless of N. If reuse distance D scales
# with N through a fixed normalized-distance law X = D/N (in this task,
# X ~ log-logistic: F(x) = x^s/(1+x^s)), then the miss rate is exactly
#   m(N) = P(D > K) = P(X > K/N) = N^s / (N^s + K^s)
# with K the PUBLIC effective capacity K = C*A/(A+1). So:
#   1. Recover s from the histogram alone, via the log-logistic's defining
#      linearity: log(F/(1-F)) = s*log(x). Rank the M samples, use the
#      empirical CDF p_i = i/(M+1) as a stand-in for F(x_i), and fit
#      logit(p_i) = a + s*log(x_i) by ordinary least squares -- the slope IS
#      the steepness, independent of any working-set size ever measured.
#   2. Compute K from the stated public formula.
#   3. Emit the closed form N^s/(N^s+K^s) with s, K baked in as constants.
# This reformulation turns curve-fitting-in-the-dark into two independently
# identifiable 1-D fits, and it is the only one of the three tiers that
# reaches past the visible flat region correctly.
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    t, C, A, M, n_train = (int(x) for x in data[:5])
    xs = [float(v) for v in data[5:5 + M]]

    xs_sorted = sorted(xs)
    logx = []
    logit = []
    for i, x in enumerate(xs_sorted, start=1):
        p = i / (M + 1.0)
        if x <= 0.0:
            continue
        logx.append(math.log(x))
        logit.append(math.log(p / (1.0 - p)))

    n = len(logx)
    sx = sum(logx); sy = sum(logit)
    sxx = sum(v * v for v in logx)
    sxy = sum(u * v for u, v in zip(logx, logit))
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        s_hat = 1.0
    else:
        s_hat = (n * sxy - sx * sy) / den
    s_hat = max(0.2, min(20.0, s_hat))

    K = C * A / (A + 1.0)
    Ks = K ** s_hat
    print("( n ** %.10g ) / ( ( n ** %.10g ) + %.10g )" % (s_hat, s_hat, Ks))


if __name__ == "__main__":
    main()
