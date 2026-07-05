# TIER: invalid
# Emits towers outside the reserve ([0,1]^d), so feasibility fails -> score 0.
import sys

def main():
    d, M = map(int, sys.stdin.read().split()[:2])
    out = []
    for i in range(M):
        out.append(" ".join("%.6f" % (2.0 + i) for _ in range(d)))
    sys.stdout.write("\n".join(out) + "\n")

main()
