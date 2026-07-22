# TIER: trivial
"""
Baseline construction: submit a single, essentially-inert rule.  This lands
close to the checker's own internal identity baseline B (predict root ==
raw word) -- an EMPTY submission is treated as infeasible (Ratio 0) by the
checker, so a minimal well-formed placeholder rule is used instead.
"""
import sys


def main():
    sys.stdin.read()  # the training corpus is not used at all
    print("0 AA A")


if __name__ == "__main__":
    main()
