# TIER: invalid
# Emits a non-finite / garbage expression -> must score 0 (feasibility gate).
import sys


def main():
    _ = sys.stdin.read()
    # 'inf' is a forbidden token AND references an undefined name -> rejected by the checker.
    print("inf * t + 1")


if __name__ == "__main__":
    main()
