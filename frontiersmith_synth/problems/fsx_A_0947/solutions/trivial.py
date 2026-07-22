# TIER: trivial
"""Do-nothing baseline: build no barriers at all. Reproduces the checker's internal
baseline exactly, so this always scores ratio ~= 0.1."""
import sys


def main():
    sys.stdin.read()  # consume input, not needed
    print(0)


if __name__ == "__main__":
    main()
