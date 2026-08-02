# TIER: trivial
# Predict "never fired" everywhere. Reproduces the checker's own internal
# all-zero baseline exactly -> Ratio ~= 0.1 by construction.
import sys


def main():
    sys.stdin.read()  # consume input (unused)
    print("0")


if __name__ == "__main__":
    main()
