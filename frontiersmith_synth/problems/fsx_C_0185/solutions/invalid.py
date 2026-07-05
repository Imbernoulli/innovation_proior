# TIER: invalid
"""Broken activation: returns a table of the WRONG length (half the grid), so the
evaluator rejects the shape and scores 0.0 on every station."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    k = int(inst["n_grid"])
    print(json.dumps([0.0] * (k // 2)))


main()
