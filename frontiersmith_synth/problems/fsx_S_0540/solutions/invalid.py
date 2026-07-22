# TIER: invalid
# Emits a non-finite / malformed expression -> grader must reject (0.0).
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("u1 + exp(u2) / 0.0 + sqrt(-1.0 - u1*u1)\n")


if __name__ == "__main__":
    main()
