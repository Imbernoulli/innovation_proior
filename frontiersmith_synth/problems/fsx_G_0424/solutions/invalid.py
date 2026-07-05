# TIER: invalid
# Emits an expression that is non-finite on the held-out points (log of a
# negative number) -> grader must reject it (0.0).
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("log(-Re) + eps\n")


if __name__ == "__main__":
    main()
