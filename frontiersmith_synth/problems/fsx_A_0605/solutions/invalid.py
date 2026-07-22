# TIER: invalid
# Emits a law that references an unknown variable, so the grader's feasibility
# check rejects it -> Ratio 0.0.
import sys


def main():
    sys.stdin.read()
    print("bogus_variable * x + 3.0")


if __name__ == "__main__":
    main()
