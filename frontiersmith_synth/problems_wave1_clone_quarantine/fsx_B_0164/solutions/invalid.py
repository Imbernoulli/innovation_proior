# TIER: invalid
# Emits a non-finite / garbage expression -> must score 0 (feasibility gate).
import sys


def main():
    _ = sys.stdin.read()
    # 'nan' is a forbidden token AND non-finite -> rejected by the checker.
    print("nan + T*V")


if __name__ == "__main__":
    main()
