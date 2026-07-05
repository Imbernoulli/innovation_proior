# TIER: invalid
# Emits a non-finite expression -> grader must reject (0.0).
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("1.0 + cos(t)/log(1.0)\n")


if __name__ == "__main__":
    main()
