# TIER: invalid
# Emits a law that references an unknown variable, so the grader's feasibility
# check rejects it -> Ratio 0.0.
import sys


def main():
    sys.stdin.read()
    print("0.5 * bogus_var + p")


if __name__ == "__main__":
    main()
