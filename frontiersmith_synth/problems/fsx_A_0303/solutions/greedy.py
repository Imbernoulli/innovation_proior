# TIER: greedy
# Single ring: place all N vials, equal radius, on one concentric ring, each as
# large as the tangency-with-neighbours and containment constraints allow.
import sys
import math

toks = sys.stdin.read().split()
N = int(toks[0])
R = float(toks[1])

s = math.sin(math.pi / N)
r = R * s / (1.0 + s)          # neighbour-tangent, contained radius
d = R - r                      # ring radius (centres this far from origin)

out = [str(N)]
for k in range(N):
    th = 2.0 * math.pi * k / N
    x = d * math.cos(th)
    y = d * math.sin(th)
    out.append("%.10f %.10f %.10f" % (x, y, r))
sys.stdout.write("\n".join(out) + "\n")
