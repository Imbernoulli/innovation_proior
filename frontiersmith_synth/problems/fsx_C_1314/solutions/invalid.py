# TIER: invalid
"""Broken cue policy: emits a cue array of the WRONG length (half the
beats), so the evaluator rejects it on every instance -> scores 0."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = int(inst["T"])
    print(json.dumps([1.0] * (T // 2)))


main()
