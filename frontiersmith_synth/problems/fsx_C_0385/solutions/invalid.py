# TIER: invalid
"""Broken imputer: emits a fill list of the WRONG length (half the holes), so the
evaluator rejects it on every garden instance -> scores 0."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    n_miss = int(inst["n_miss"])
    print(json.dumps([0.0] * (n_miss // 2)))


main()
