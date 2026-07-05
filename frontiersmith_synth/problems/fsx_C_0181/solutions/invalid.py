# TIER: invalid
"""Broken detector: emits a score array of the WRONG length (half the rows), so
the evaluator rejects it on every fleet -> scores 0."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    n = int(inst["n"])
    print(json.dumps([0.0] * (n // 2)))


main()
