# TIER: invalid
# Emits a non-finite incidence expression -> grader must reject (0.0).
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("x1 + exp(30.0 * x1) / (x1 - x1)\n")


if __name__ == "__main__":
    main()
