# TIER: invalid
# Emits N maximal circles all stacked at the plot centre: every pair overlaps and
# each pokes past the plot walls, so the feasibility gate rejects it -> score 0.
import sys


def read_instance():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    W = float(toks[1])
    H = float(toks[2])
    return N, W, H


N, W, H = read_instance()
R = max(W, H)
out = [str(N)]
for _ in range(N):
    out.append("%.10f %.10f %.10f" % (W / 2.0, H / 2.0, R))
sys.stdout.write("\n".join(out) + "\n")
