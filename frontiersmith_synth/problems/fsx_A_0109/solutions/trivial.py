# TIER: trivial
# Corner grid: probes at (i%g)/g, (i//g)/g -- clustered toward origin.
# This reproduces the checker's internal baseline -> ratio ~0.1.
import sys, math


def main():
    toks = sys.stdin.read().split()
    d = int(toks[0]); M = int(toks[1]); K = int(toks[2])
    nf = M - K
    g = int(math.ceil(math.sqrt(nf)))
    out = []
    for i in range(nf):
        out.append("%.17g %.17g" % ((i % g) / g, (i // g) / g))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
