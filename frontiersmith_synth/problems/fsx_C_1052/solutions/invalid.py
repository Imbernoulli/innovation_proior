# TIER: invalid
#
# Emits a word that uses a digit equal to the alphabet size itself, i.e. a
# character outside the valid range {0, ..., a-1}. Must be rejected by the
# checker's feasibility check -> Ratio: 0.0.
import sys


def main():
    a, p, L = map(int, sys.stdin.read().split()[:3])
    bad_digit = str(a % 10)  # guaranteed >= a, out of the valid alphabet
    sys.stdout.write(bad_digit * max(1, min(L, 10)) + "\n")


if __name__ == "__main__":
    main()
