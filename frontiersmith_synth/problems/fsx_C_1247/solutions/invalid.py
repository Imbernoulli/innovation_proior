# TIER: invalid
"""
Emits zero refresh events. Every weak row's retention bound is far smaller than the horizon,
so this deterministically violates the retention constraint everywhere -> Ratio: 0.0.
"""
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("0\n")


if __name__ == "__main__":
    main()
