# TIER: trivial
# Naive half-fill: ping uniformly across the first half of the passage, stay silent after.
# This reproduces the checker's internal baseline B -> Ratio ~ 0.1.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    L = n // 2
    f = [1.0] * L + [0.0] * (n - L)
    sys.stdout.write(" ".join("%.6f" % x for x in f) + "\n")

if __name__ == "__main__":
    main()
