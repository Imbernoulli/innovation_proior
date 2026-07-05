# TIER: trivial
# Arithmetic-progression apiary: boxes at 0,1,...,n-1.
# Reproduces the checker's baseline exactly -> Ratio 0.1 on every case.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    print(" ".join(str(x) for x in range(n)))


if __name__ == "__main__":
    main()
