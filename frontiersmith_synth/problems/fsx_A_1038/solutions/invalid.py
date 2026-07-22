# TIER: invalid
# Emits a duplicate site (schema violation) -- must score exactly 0.
import sys


def main():
    sys.stdin.read()
    print(2)
    print(0, 0)
    print(0, 1)


main()
