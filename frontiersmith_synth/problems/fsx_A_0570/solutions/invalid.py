# TIER: invalid
# Emits a syntactically valid but incorrect chain: one doubling of the base,
# producing only the value 2.  It does not produce the targets -> Ratio 0.0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("1\n0 0\n")


if __name__ == "__main__":
    main()
