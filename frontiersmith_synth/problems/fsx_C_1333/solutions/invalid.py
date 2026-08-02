# TIER: invalid
# Emits a mix with wildly excessive water and cement -- violates the water/cement
# ratio bound, the aggregate-volume floor and (likely) the shrinkage budget too.
import sys


def main():
    sys.stdin.read()  # drain input, ignore it
    print(1, 10000.0, 10000.0, 0.0)


if __name__ == "__main__":
    main()
