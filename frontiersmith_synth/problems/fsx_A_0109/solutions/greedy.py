# TIER: greedy
# Centered grid: probes at cell centers ((i%g)+0.5)/g, ((i//g)+0.5)/g.
# A genuine 2-D lattice -> lower star discrepancy than the corner grid.
import sys, math


def main():
    toks = sys.stdin.read().split()
    d = int(toks[0]); M = int(toks[1]); K = int(toks[2])
    nf = M - K
    g = int(math.ceil(math.sqrt(nf)))
    out = []
    for i in range(nf):
        out.append("%.17g %.17g" % (((i % g) + 0.5) / g, ((i // g) + 0.5) / g))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
