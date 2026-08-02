# TIER: invalid
# References an undeclared variable -> the checker's grammar check must
# reject this and score exactly 0.
import sys


def main():
    sys.stdin.read()
    print("0.5*y0 + 3.0*q_unknown_tap")


if __name__ == "__main__":
    main()
