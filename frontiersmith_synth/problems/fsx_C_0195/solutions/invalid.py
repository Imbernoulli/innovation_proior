# TIER: invalid
"""Malformed schedule: 'eta' is a string, and the step arrays have the wrong
length.  The evaluator's strict validation rejects it, so every instance scores
exactly 0."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = int(inst["T"])
    # wrong types / wrong length on purpose
    print(json.dumps({"eta": "fast", "alpha": [1.0] * (T + 3), "beta": 0.0}))


main()
