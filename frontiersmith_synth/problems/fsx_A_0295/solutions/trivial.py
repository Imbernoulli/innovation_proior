# TIER: trivial
# Half-block: cram unit capacity into the first floor(n/2) reservoirs, zero
# elsewhere. This is exactly the checker's internal baseline, so it scores ~0.1.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    f = [1.0] * (n // 2) + [0.0] * (n - n // 2)
    print(" ".join("%.6f" % x for x in f))


if __name__ == "__main__":
    main()
