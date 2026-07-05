# TIER: invalid
"""Broken classifier: returns a label list of the WRONG length (half the
queries), so the evaluator rejects it on every instance -> scores 0."""
import sys
import json


def main():
    inst = json.load(sys.stdin)
    q = len(inst["queries"])
    print(json.dumps([0] * (q // 2)))


main()
