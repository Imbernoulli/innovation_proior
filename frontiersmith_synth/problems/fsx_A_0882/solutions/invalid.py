# TIER: invalid
"""
Invalid: emits a law that produces a non-finite value (division by x-x, a
guaranteed zero) -- must score 0 under the checker's finiteness/positivity
gate.
"""
import sys


def main():
    sys.stdin.read()
    print("1.0 / (x - x)")


if __name__ == "__main__":
    main()
