# TIER: trivial
"""
Inline nothing. Every call site pays its full base cost plus call overhead. This is the
textbook "do nothing" baseline -- cheap to write, leaves every removable cost on the
table and never touches the icache budget at all.
"""
import sys


def main():
    sys.stdin.read()  # instance is unused
    print(0)
    print()


if __name__ == "__main__":
    main()
