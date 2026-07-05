# TIER: invalid
# Emits overlapping, out-of-box circles with an oversized radius -> infeasible -> scores 0.
import sys

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); W = float(toks[1]); H = float(toks[2]); rmax = float(toks[3])
    # All circles at the same point with radius > rmax and larger than the box:
    # violates the radius cap, containment, and non-overlap all at once.
    out = []
    for _ in range(N):
        out.append("%.6f %.6f %.6f" % (W * 0.5, H * 0.5, max(W, H) * 2.0 + rmax))
    sys.stdout.write("\n".join(out) + "\n")

main()
