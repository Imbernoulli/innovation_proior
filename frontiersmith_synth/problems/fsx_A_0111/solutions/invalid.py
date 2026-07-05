# TIER: invalid
# Emits an infeasible artifact: the three pads (0,0),(1,0),(0,1) which are always
# unobstructed but form a corner (d=1) -> must score 0.0.
import sys


def main():
    sys.stdin.read()
    print(3)
    print("0 0")
    print("1 0")
    print("0 1")


if __name__ == "__main__":
    main()
