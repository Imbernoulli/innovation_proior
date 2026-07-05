# TIER: invalid
# Emits an expression referencing an undeclared variable (x9) -> the checker's
# AST whitelist rejects it -> Ratio 0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("3.0*x9 + x0*x1 - 2.0\n")


if __name__ == "__main__":
    main()
