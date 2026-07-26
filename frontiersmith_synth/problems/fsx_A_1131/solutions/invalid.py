# TIER: invalid
# Emits an expression referencing an unknown function name -> the checker's
# strict AST validator rejects it and prints Ratio: 0.0.
import sys


def main():
    print("EXPR 100 + magic(t) * 3")


if __name__ == "__main__":
    main()
