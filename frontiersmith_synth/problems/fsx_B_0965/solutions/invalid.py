# TIER: invalid
# Emits a syntactically legal MODE line but an expression that blows up at evaluation time
# (division by zero on every withheld tag) -> the checker's feasibility check must catch this
# and score 0.
import sys


def main():
    _ = sys.stdin.read()
    print("MODE N")
    print("1 / (n - n)")


if __name__ == "__main__":
    main()
