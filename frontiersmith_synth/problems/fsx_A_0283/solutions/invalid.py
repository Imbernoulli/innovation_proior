# TIER: invalid
# Emits an infeasible placement: N heavily overlapping oversized pads stacked at the centre
# (and colliding with the forbidden zones). Must score 0 under strict feasibility checks.
import sys

t = sys.stdin.read().split()
N = int(t[0]); S = float(t[1])
out = [str(N)]
for _ in range(N):
    out.append("%.6f %.6f %.6f" % (S / 2.0, S / 2.0, 0.45))
sys.stdout.write("\n".join(out) + "\n")
