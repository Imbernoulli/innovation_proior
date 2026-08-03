# TIER: greedy
# Simplified single-exponent scaling law: assume turnaround depends only on the
# work-per-crane ratio, T ~ a * (n/c)^b, and fit a, b in log-log space. This
# couples the workload and crane exponents (forces them equal), so it captures the
# gross trend but mis-extrapolates because the true law scales n and c differently.
import sys, math
d = sys.stdin.read().split()
M = int(d[0])
xs, ys = [], []
for i in range(M):
    n = float(d[1 + 3 * i]); c = float(d[2 + 3 * i]); t = float(d[3 + 3 * i])
    xs.append(math.log(n / c)); ys.append(math.log(t))
mx = sum(xs) / M; my = sum(ys) / M
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
a = math.exp(my - b * mx)
print("%.6f * (n / c) ** %.6f" % (a, b))
