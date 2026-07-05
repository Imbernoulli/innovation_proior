# TIER: invalid
# Emit an expression that blows up (division by zero) when evaluated on the
# held-out frequencies -> grader must reject with Ratio 0.0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("1.0/(f-f)\n")


if __name__ == "__main__":
    main()
