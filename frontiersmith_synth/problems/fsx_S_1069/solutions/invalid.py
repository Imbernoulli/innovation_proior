# TIER: invalid
# Emits a syntactically well-formed but wrong program: it just computes the
# constant 0, which never equals P(x) for our instances (a_n != 0, degree
# >= 4) -> checker's exact-equivalence gate rejects it -> Ratio 0.0.
import sys


def main():
    sys.stdin.read()  # consume input, unused
    sys.stdout.write("1\nC 0 1\n")


if __name__ == "__main__":
    main()
