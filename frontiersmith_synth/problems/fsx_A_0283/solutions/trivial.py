# TIER: trivial
# Bottom-row baseline: N equal pads laid along the clear bottom margin y = r, radius
# r = S/(2N). Total F = S/2 = B, so this reproduces the checker baseline (~0.1). Because
# every forbidden zone is inside [0.15,0.85]^2 and 2r <= 2*S/16 = 0.125 < 0.15 for N >= 8,
# the whole row is guaranteed clear of the zones.
import sys

t = sys.stdin.read().split()
N = int(t[0]); S = float(t[1])
r = (S / (2.0 * N)) * (1.0 - 1e-7)  # tiny shrink to stay strictly feasible under FP
out = [str(N)]
for i in range(N):
    x = (i + 0.5) * S / N
    y = S / (2.0 * N)
    out.append("%.10f %.10f %.10f" % (x, y, r))
sys.stdout.write("\n".join(out) + "\n")
