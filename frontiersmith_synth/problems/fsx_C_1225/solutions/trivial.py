# TIER: trivial
"""Do-nothing baseline: one shared, non-isolating bucket for every key.
This reproduces the checker's own internal baseline construction exactly."""
import sys


def main():
    sys.stdin.read()  # trace is ignored -- no per-key structure at all
    print("0 1")
    print("20 3")


if __name__ == "__main__":
    main()
