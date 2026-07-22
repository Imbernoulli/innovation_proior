# TIER: invalid
import sys


def main():
    d = sys.stdin.read().split()
    a = int(d[0]); k = int(d[1]); L = int(d[2])
    # Emit a string of the right length but built from a digit equal to the alphabet
    # size itself (a), which is out of the valid range [0, a) -- infeasible output.
    print(str(a) * L)


if __name__ == "__main__":
    main()
