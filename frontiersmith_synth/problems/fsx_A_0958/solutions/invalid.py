# TIER: invalid
# Emits an expression using a disallowed token (attribute access) -> Ratio 0.0.
import sys


def main():
    sys.stdin.read()
    print("f.bit_length() + a")


if __name__ == "__main__":
    main()
