# TIER: invalid
# Emits a non-finite / malformed expression -> grader must reject (0.0).
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("q1 + exp(nan)\n")


if __name__ == "__main__":
    main()
