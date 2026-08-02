# TIER: invalid
"""Malformed artifact: unknown opcode and a missing RESULT line. Must score
0.0 under strict structural validation."""
import sys


def main():
    sys.stdin.read()
    print("FOO r0 r1 r2")
    print("r16 #999999999999 #1")


if __name__ == "__main__":
    main()
