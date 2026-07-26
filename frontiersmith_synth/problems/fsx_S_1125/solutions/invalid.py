# TIER: invalid
# Emits an expression that raises at evaluation time (division by zero) ->
# grader must reject (0.0).
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("t/0.0 + p + exp(t)\n")


if __name__ == "__main__":
    main()
