# TIER: invalid
import sys


def main():
    sys.stdin.read()  # ignore input entirely
    # infeasible: 8 contracts (exceeds the K<=6 cap), and one coverage value
    # is out of range -- must score 0 under strict feasibility checking.
    print(8)
    for _ in range(7):
        print("100.0 0.5")
    print("50.0 1.7")


if __name__ == "__main__":
    main()
