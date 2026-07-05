# TIER: greedy
# Temperature-only Arrhenius fit: assume the rate depends only on temperature,
# ln k ~ ln A - E/T, and fit (ln A, E) by least squares in the feature 1/T. This
# captures the dominant exponential temperature trend but IGNORES the concentration
# order C**n, so it systematically mis-extrapolates into the hot, concentrated regime.
import sys, math
d = sys.stdin.read().split()
M = int(d[0])
xs, ys = [], []
for i in range(M):
    T = float(d[1 + 3 * i]); k = float(d[3 + 3 * i])
    xs.append(1.0 / T)
    ys.append(math.log(k))
mx = sum(xs) / M; my = sum(ys) / M
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
intercept = my - slope * mx
A = math.exp(intercept)     # slope ~ -E
print("%.10e * exp(%.8f / T)" % (A, slope))
