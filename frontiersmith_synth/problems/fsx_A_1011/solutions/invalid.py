# TIER: invalid
# Schedules a single operation at a start time far past the instance's horizon,
# so the checker's horizon/feasibility gate rejects the whole submission -> 0.
import sys


def main():
    sys.stdin.read()  # consume the instance
    print(1)
    print("OP 1 999999999.0")


if __name__ == "__main__":
    main()
