# TIER: trivial
# Arithmetic run 0..n-1 -- reproduces the checker's internal baseline (score ~0.1).
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0]); M = int(data[1])
    A = list(range(n))          # always fits since M >= n-1
    out = [str(n)] + [str(x) for x in A]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
