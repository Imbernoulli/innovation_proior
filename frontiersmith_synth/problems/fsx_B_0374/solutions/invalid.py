# TIER: invalid
"""Emits a garbage expression that references a disallowed name -> scores 0."""
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("nan + x1*__import__('os') + zzz\n")


if __name__ == "__main__":
    main()
