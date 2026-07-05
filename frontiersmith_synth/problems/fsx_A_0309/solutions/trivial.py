# TIER: trivial
# Ranger-diagonal placement: every tower on the main diagonal of the reserve.
# This reproduces the checker's internal baseline, so it scores ~0.1.
import sys

def main():
    d, M = map(int, sys.stdin.read().split()[:2])
    out = []
    for i in range(M):
        c = (i + 0.5) / M
        out.append(" ".join("%.12f" % c for _ in range(d)))
    sys.stdout.write("\n".join(out) + "\n")

main()
