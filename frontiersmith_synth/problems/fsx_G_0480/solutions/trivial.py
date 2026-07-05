# TIER: trivial
# Baseline conflict-free code: the powers of two <= n. Reproduces the checker's
# internal baseline B, so it scores ~0.1.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    out = []
    x = 1
    while x <= n:
        out.append(x)
        x *= 2
    sys.stdout.write(" ".join(map(str, out)) + "\n")


if __name__ == "__main__":
    main()
