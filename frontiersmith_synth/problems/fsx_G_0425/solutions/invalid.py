# TIER: invalid
"""
Invalid: emits a non-finite / disallowed expression. Must score 0.
"""
import sys


def main():
    sys.stdin.read()
    # references a forbidden variable and blows up to inf -> rejected
    print("exp(exp(exp(T))) + Z/0.0")


if __name__ == "__main__":
    main()
