# TIER: trivial
# Baseline construction: realize every tap as an exact 1/2 divider from two matched
# units of catalog value[0]. Corner-invariant but ignores the targets -> reproduces
# the checker's internal baseline B (score ~0.1).
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    K = int(next(it)); M = int(next(it))
    lines = []
    for _ in range(K):
        top = [0] * M; bot = [0] * M
        top[0] = 1; bot[0] = 1
        lines.append(" ".join(map(str, top + bot)))
    sys.stdout.write("\n".join(lines) + "\n")

main()
