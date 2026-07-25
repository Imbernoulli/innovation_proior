# TIER: trivial
# Naive direct probes: measure the first m coordinates at full amplitude.
# Reproduces the checker's internal baseline construction (scores ~0.1).
import sys


def main():
    data = sys.stdin.read().split()
    n, m, K, s, pmax = (int(data[i]) for i in range(5))
    out = []
    for j in range(m):
        row = ["0"] * n
        row[j % n] = str(pmax)
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


main()
