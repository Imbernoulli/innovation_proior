# TIER: trivial
# Single-row packing -- reproduces the checker baseline (~0.1).
# N equal circles along the horizontal midline, spaced W/N apart.
import sys

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); W = float(toks[1]); H = float(toks[2]); rmax = float(toks[3])
    r = min(0.5 * W / N, 0.5 * H, rmax)
    cw = W / N
    out = []
    for i in range(N):
        x = (i + 0.5) * cw
        y = 0.5 * H
        out.append("%.9f %.9f %.9f" % (x, y, r))
    sys.stdout.write("\n".join(out) + "\n")

main()
