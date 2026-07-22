# TIER: invalid
# Emits a feedback exponent far outside the allowed range [0.2, 4.0] ->
# fails the mu-range feasibility check -> scores 0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("99.0\n0\n")


if __name__ == "__main__":
    main()
