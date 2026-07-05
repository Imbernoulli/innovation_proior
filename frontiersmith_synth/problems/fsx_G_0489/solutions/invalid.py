# TIER: invalid
# Emits the full interval [0, W]: distinct, in range, diameter exactly W -- but NOT
# admissible (mod 2 it occupies both residue classes), so the checker must score it 0.
import sys


def main():
    W = int(sys.stdin.read().split()[0])
    print(" ".join(map(str, range(W + 1))))


main()
