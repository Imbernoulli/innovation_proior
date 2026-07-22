# TIER: invalid
"""
Invalid: emits a law that produces a non-finite value (division by a T-T
zero) -- must score 0 under the checker's finiteness/positivity gate.
"""
import sys


def main():
    sys.stdin.read()
    print("1.0 / (T - T)")


if __name__ == "__main__":
    main()
