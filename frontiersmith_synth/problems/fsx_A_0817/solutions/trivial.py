# TIER: trivial
"""Reproduces the checker's own internal baseline construction exactly: a pure
translation generator along +x, with no attempt to read the data at all."""
import sys


def main():
    sys.stdin.read()  # consume input (ignored)
    print("0 0 0 0 1 0")


if __name__ == "__main__":
    main()
