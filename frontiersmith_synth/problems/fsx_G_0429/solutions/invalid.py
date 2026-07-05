# TIER: invalid
# Emits a non-finite expression -> the grader must reject it (0.0).
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("x1*x4 - x2*x3 + log(x3 - x3)\n")


if __name__ == "__main__":
    main()
