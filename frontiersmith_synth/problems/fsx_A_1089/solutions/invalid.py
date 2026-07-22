# TIER: invalid
#
# Invalid reference: emits a well-formed but WRONG circuit (constant 0).
# The language of every generated instance is non-empty, so the exact
# truth-table equivalence gate must reject this with Ratio 0.0.

import sys


def main():
    n = int(sys.stdin.read().split()[0])
    sys.stdout.write("1\nCONST 0\nOUT %d\n" % n)


if __name__ == "__main__":
    main()
