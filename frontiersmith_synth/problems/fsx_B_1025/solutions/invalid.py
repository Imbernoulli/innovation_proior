# TIER: invalid
# Emits an out-of-range, garbage artifact: gains far above the stability bound
# (>1 in magnitude) and one non-finite delay-length token. Must score 0.0.
import sys


def main():
    tok = sys.stdin.read().split()
    N = int(tok[0])
    L = ["nan"] + [str(10**9) for _ in range(N - 1)]
    g = ["1.5"] * N
    out = []
    out.append(" ".join(L))
    out.append(" ".join(g))
    sys.stdout.write("\n".join(out) + "\n")


main()
