# TIER: invalid
# Emits an infeasible artifact: a string containing the character 'z', which
# is never part of the (<=3-symbol, drawn from {a,b,c}) alphabet -> the
# checker must reject it with Ratio 0.0 regardless of the instance.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("1\nzzz\n")


if __name__ == "__main__":
    main()
