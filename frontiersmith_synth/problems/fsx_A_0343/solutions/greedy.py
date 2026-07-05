# TIER: greedy
# Equal-radius 2D grid packing capped at rmax. Uses both dimensions of the panel
# instead of a single row -> substantially larger total than the trivial baseline.
import sys, math

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); W = float(toks[1]); H = float(toks[2]); rmax = float(toks[3])
    G = int(math.ceil(math.sqrt(N)))
    R = int(math.ceil(N / float(G)))
    cw = W / G
    ch = H / R
    r = min(0.49 * min(cw, ch), rmax)
    out = []
    k = 0
    for row in range(R):
        for col in range(G):
            if k >= N:
                break
            x = (col + 0.5) * cw
            y = (row + 0.5) * ch
            out.append("%.9f %.9f %.9f" % (x, y, r))
            k += 1
    sys.stdout.write("\n".join(out) + "\n")

main()
