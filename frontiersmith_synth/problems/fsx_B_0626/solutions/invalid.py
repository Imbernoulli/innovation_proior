# TIER: invalid
import sys

# Emits a well-formed but functionally wrong circuit: zero gates, i.e. the identity map.
# Every instance in this family plants a non-identity sigma (the generator rejects any
# sigma with accidental symmetry, which rules out the identity permutation itself), so
# pi is never the identity and this always fails the checker's exact-equivalence gate.


def main():
    sys.stdin.read()
    print(0)


if __name__ == "__main__":
    main()
