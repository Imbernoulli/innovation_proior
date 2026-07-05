# TIER: greedy
# Uniform flat sweep density: f_i = 1 for all i. This gives c1 = 2 exactly,
# a solid improvement over the smooth-bump baseline but far from optimal.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    print(" ".join(["1.0"] * n))


if __name__ == "__main__":
    main()
