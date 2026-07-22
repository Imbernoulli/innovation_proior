# TIER: invalid
# Infeasible output: references a disallowed name inside REC and mismatches
# its own BASE count -- must score 0 under strict grammar validation.
import sys


def main():
    sys.stdin.read()
    print("BASE 2 1 1")          # claims k=2 (3 values) but only gives 2
    print("REC n * secret_growth_rate")


if __name__ == "__main__":
    main()
