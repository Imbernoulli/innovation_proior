# TIER: invalid
"""Emits a disallowed/garbage artifact -- must score 0."""
import sys


def main():
    sys.stdin.read()
    print("L0 + nan*O/(Cap-O)")  # 'nan' is not an allowed name -> parse rejects it


if __name__ == "__main__":
    main()
