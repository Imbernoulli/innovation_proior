# TIER: invalid
"""Out-of-range codes on every line (only 0..5 are legal token codes) ->
the checker must reject this outright -> Ratio 0."""
import sys


def main():
    sys.stdin.read()
    print("99")
    print("99 98")
    print("77")


if __name__ == "__main__":
    main()
