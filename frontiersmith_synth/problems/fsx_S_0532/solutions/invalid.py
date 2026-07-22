# TIER: invalid
# Emits an expression referencing an undeclared variable (u) -> the checker's AST
# whitelist rejects it -> Ratio 0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("1.5*u**4 - 2.0*u**3 + u\n")


if __name__ == "__main__":
    main()
