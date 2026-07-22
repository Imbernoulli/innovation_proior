# TIER: trivial
# Baseline: clean before every job (id order). Reproduces the checker's reference -> ~0.1.
import sys

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    out = []
    for j in range(N):
        if j > 0:
            out.append("C")
        out.append(str(j))
    sys.stdout.write(" ".join(out) + "\n")

main()
