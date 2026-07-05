# TIER: greedy
# Single-variable power law: assume life-support power depends only on habitat
# volume, P ~ a * V^b, fit a, b in log-log space. Ignores crew size entirely, so
# it captures the volume trend but mis-extrapolates because large surface bases
# also carry much larger crews (the dominant superlinear driver).
import sys, math
d = sys.stdin.read().split()
M = int(d[0])
xs, ys = [], []
for i in range(M):
    V = float(d[2 + 3 * i]); p = float(d[3 + 3 * i])
    xs.append(math.log(V)); ys.append(math.log(p))
mx = sum(xs) / M; my = sum(ys) / M
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
a = math.exp(my - b * mx)
print("%.6f * V ** %.6f" % (a, b))
