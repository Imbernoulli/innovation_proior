# TIER: trivial
# Output the checker's own un-tuned baseline linkage (0.35g, 1.10g, 0.90g,
# u=0.5b, v=0.3b) derived from the ground span alone -> reproduces baseline ~0.1.
import sys, math

toks = sys.stdin.read().split()
M = int(toks[0])
O0 = (float(toks[1]), float(toks[2]))
O1 = (float(toks[3]), float(toks[4]))
g = math.hypot(O1[0] - O0[0], O1[1] - O0[1])

a, b, c = 0.35 * g, 1.10 * g, 0.90 * g
u, v = 0.50 * b, 0.30 * b
print("%.10f %.10f %.10f %.10f %.10f" % (a, b, c, u, v))
