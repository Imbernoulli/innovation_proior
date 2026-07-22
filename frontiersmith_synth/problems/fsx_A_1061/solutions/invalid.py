# TIER: invalid
# Emits k copies of channel 0 -> fails the distinctness check -> scores 0.
import sys


def main():
    toks = sys.stdin.read().split()
    k = int(toks[2])
    sys.stdout.write(" ".join(["0"] * k) + "\n")


if __name__ == "__main__":
    main()
