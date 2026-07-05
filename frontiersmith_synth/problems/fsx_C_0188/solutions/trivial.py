# TIER: trivial
"""Trivial baseline: return the EMPTY flow graph (claim no causal edges at all).

This reproduces the evaluator's internal construction, whose SHD equals the number
of true edges E, so every grid normalizes to ~0.1.  A no-op that establishes the
floor of the ladder."""
import sys, json


def main():
    json.load(sys.stdin)          # consume the public instance
    print(json.dumps({"edges": []}))


main()
