# TIER: invalid
import sys

def main():
    sys.stdin.read()  # ignore the instance entirely
    # a single interval with a negative start tick: always out of [0, T),
    # guaranteed infeasible regardless of the instance parameters.
    print(1)
    print("0 0 -5 5")

if __name__ == "__main__":
    main()
