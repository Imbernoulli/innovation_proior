# TIER: trivial
"""Predict the EMPTY visitor-flow map: no edges at all.  Its SHD equals the number
of true edges, which is exactly the evaluator's empty-graph baseline, so every
museum normalizes to ~0.1.  A valid, reproducible, do-nothing lower bound."""
import sys, json


def main():
    json.load(sys.stdin)          # consume the public instance
    print(json.dumps({"edges": []}))


main()
