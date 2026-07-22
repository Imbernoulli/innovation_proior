# TIER: invalid
# Emits only re-dress tokens: every real job is missing, so the schedule is
# infeasible and the checker must score it 0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("0 0 0")


if __name__ == "__main__":
    main()
