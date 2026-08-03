# TIER: invalid
"""Broken schedule: returns a list of the WRONG length (half the epoch budget), so
the evaluator rejects the shape and scores 0.0 on every rooftop plot."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    E = int(inst["n_epochs"])
    print(json.dumps([1.0] * (E // 2)))


main()
