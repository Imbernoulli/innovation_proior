# TIER: invalid
# Emits a non-finite / garbage expression -> must score 0 (feasibility gate).
import sys


def main():
    _ = sys.stdin.read()
    # 'inf' is a forbidden token AND the expression blows up -> rejected by the checker.
    print("1e9 * inf + s / (r - r)")


if __name__ == "__main__":
    main()
