# TIER: trivial
# Reproduce the grader's internal baseline: the fixed "no discount" predictor f(n) = n.
# It ignores the training data entirely, so it is exact only on the (rare) tags whose
# multiplicative discount happens to be zero -> Ratio ~= 0.10 by construction (this IS the
# checker's own baseline construction B).
import sys


def main():
    _ = sys.stdin.read()
    print("MODE N")
    print("n")


if __name__ == "__main__":
    main()
