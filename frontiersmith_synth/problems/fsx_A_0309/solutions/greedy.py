# TIER: greedy
# Irrational rank-1 Kronecker lattice: tower i sits at frac(i * alpha_k) per axis.
# Quasi-uniform, clearly beats the diagonal, but leaves plenty of headroom.
import sys
import math

ALPHAS = [math.sqrt(2.0), math.sqrt(3.0), math.sqrt(5.0)]

def main():
    d, M = map(int, sys.stdin.read().split()[:2])
    out = []
    for i in range(1, M + 1):
        coords = []
        for k in range(d):
            v = (i * ALPHAS[k]) % 1.0
            coords.append("%.12f" % v)
        out.append(" ".join(coords))
    sys.stdout.write("\n".join(out) + "\n")

main()
