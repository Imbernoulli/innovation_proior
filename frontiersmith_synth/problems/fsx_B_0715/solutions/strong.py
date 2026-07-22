# TIER: strong
# Insight: the hypothesis class is a LATTICE, not a manifold. Individually
# each calibration brew gives only weak, noisy evidence about the hidden
# stoichiometry s_i (all reagents sit near a common balanced batch scale).
# But if we SNAP candidate ratios to small coprime integers and let least
# squares aggregate the evidence over all brews, the fit sharply favours the
# true integer vector: any wrong s' reshuffles which reagent looks
# "limiting" at different (batch-scale, jitter) draws, which the true s does
# not, so its training SSE is systematically higher.
#
# Enumerate coprime s in {1..6}^4, fit the single gain g in closed form
# (least squares against m(s) = min_i(q_i/s_i), no intercept -- the true law
# has none), keep the s with smallest training SSE, and emit the exact
# min-law expression with the recovered integers plugged in.
import sys, math


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    Q = []; Y = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 5:
            v = list(map(float, p[:5]))
            Q.append(v[:4]); Y.append(v[4])

    S_MAX = 6
    best = None  # (sse, s, g)
    for s1 in range(1, S_MAX + 1):
        for s2 in range(1, S_MAX + 1):
            for s3 in range(1, S_MAX + 1):
                for s4 in range(1, S_MAX + 1):
                    g0 = math.gcd(math.gcd(s1, s2), math.gcd(s3, s4))
                    if g0 != 1:
                        continue
                    s = (s1, s2, s3, s4)
                    num = 0.0
                    den = 0.0
                    for q, y in zip(Q, Y):
                        m = min(q[0] / s1, q[1] / s2, q[2] / s3, q[3] / s4)
                        num += y * m
                        den += m * m
                    if den <= 1e-12:
                        continue
                    g = num / den
                    sse = 0.0
                    for q, y in zip(Q, Y):
                        m = min(q[0] / s1, q[1] / s2, q[2] / s3, q[3] / s4)
                        d = y - g * m
                        sse += d * d
                    if best is None or sse < best[0]:
                        best = (sse, s, g)

    _, s, g = best
    print("%r * min(q1/%d, q2/%d, q3/%d, q4/%d)" % (g, s[0], s[1], s[2], s[3]))


if __name__ == "__main__":
    main()
