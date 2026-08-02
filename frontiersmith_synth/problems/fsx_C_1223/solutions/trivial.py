# TIER: trivial
"""Empty sync set for every context: on the first unexpected token in a
program, skip straight to end-of-input (no resynchronization at all).
Reproduces the checker's own internal baseline B exactly -> Ratio ~ 0.1."""
import sys


def main():
    sys.stdin.read()  # instance content is irrelevant to this fixed policy
    print("")
    print("")
    print("")


if __name__ == "__main__":
    main()
