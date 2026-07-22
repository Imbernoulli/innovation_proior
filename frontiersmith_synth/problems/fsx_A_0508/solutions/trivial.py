# TIER: trivial
# Do nothing: pour 0 grains every stage. Reproduces the checker's baseline B.
import sys


def main():
    d = sys.stdin.read().split()
    N = int(d[0]); K = int(d[1])
    print("\n".join("0 0" for _ in range(K)))


main()
