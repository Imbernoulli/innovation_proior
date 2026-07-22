# TIER: invalid
# Emits a well-formed but useless program (produces only g^2); none of the large
# targets are produced -> checker scores 0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("1\n0 0\n")


if __name__ == "__main__":
    main()
