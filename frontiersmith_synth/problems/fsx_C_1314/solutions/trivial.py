# TIER: trivial
"""Never move: broadcast a flat cue of 1.0 for the whole passage. This
reproduces the evaluator's internal baseline exactly, so every instance
normalizes to ~0.1."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = int(inst["T"])
    print(json.dumps([1.0] * T))


main()
