# TIER: invalid
"""
Invalid: emits a law that divides by (t - t), i.e. zero, at every evaluation
point -- must score 0 under the checker's finiteness/positivity gate.
"""
import sys


def main():
    sys.stdin.read()
    print("1.0 / (t - t)")


if __name__ == "__main__":
    main()
