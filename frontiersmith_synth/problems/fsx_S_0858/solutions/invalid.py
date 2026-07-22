# TIER: invalid
# Emits a syntactically well-formed but WRONG program: the identity function
# (register 0 unmodified via a no-op XOR with 0). Since none of the planted
# targets are the identity, this reconstructs the wrong table everywhere ->
# checker scores 0.
import sys


def main():
    data = sys.stdin.read().split()
    w = int(data[0])
    n = 1 << w
    # ignore the table entirely; just emit "y = x XOR 0"
    sys.stdout.write("2\nCONST 0\nXOR 0 1\n")


if __name__ == "__main__":
    main()
