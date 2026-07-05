# TIER: invalid
# Emits a non-finite implicit relation -> the grader must reject it (0.0).
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("x*y + sqrt(x**2 + y**2) - inf\n")


if __name__ == "__main__":
    main()
