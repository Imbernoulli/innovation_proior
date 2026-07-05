# TIER: invalid
# Emits negative ping energies -> violates the 0 <= f[i] feasibility rule -> Ratio 0.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    f = [-1.0] * n
    sys.stdout.write(" ".join("%.6f" % x for x in f) + "\n")

if __name__ == "__main__":
    main()
