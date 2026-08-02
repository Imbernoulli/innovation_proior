# TIER: invalid
# Deliberately infeasible: emits a fragment id far outside the valid range
# [0, M-1]. Must score 0 regardless of instance.
import sys


def main():
    sys.stdin.read()  # instance is irrelevant to this deliberately-broken output
    print(5)
    print("0 0 0 0 999999")


if __name__ == "__main__":
    main()
