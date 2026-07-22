# TIER: invalid
# Emits an expression that references a disallowed name/function -> the
# checker's strict grammar validator rejects it and prints Ratio: 0.0.
import sys


def main():
    sys.stdin.read()
    print("exp(n) + import_os(n)")


if __name__ == "__main__":
    main()
