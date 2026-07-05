# TIER: invalid
"""Broken clustering: emits a label list of the WRONG length (half the rows), so
the evaluator rejects it on every resort instance -> scores 0."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    n = int(inst["n"])
    print(json.dumps([0] * (n // 2)))


main()
