# TIER: invalid
# Emits N maximal vials all stacked at the centre: every pair overlaps, so the
# feasibility gate rejects it and the score must be 0.
import sys

toks = sys.stdin.read().split()
N = int(toks[0])
R = float(toks[1])

out = [str(N)]
for _ in range(N):
    out.append("%.10f %.10f %.10f" % (0.0, 0.0, R))
sys.stdout.write("\n".join(out) + "\n")
