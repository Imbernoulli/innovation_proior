# TIER: trivial
"""Majority-label predictor: emit the most common TRAIN label for every query.
This reproduces the evaluator's internal baseline exactly, so every instance
normalizes to ~0.1.  It never looks at the query missions at all -> no
generalization, no rule induction."""
import sys
import json


def main():
    inst = json.load(sys.stdin)
    train = inst["train"]
    ones = sum(1 for r in train if r["label"] == 1)
    zeros = len(train) - ones
    maj = 1 if ones > zeros else 0
    print(json.dumps([maj] * len(inst["queries"])))


main()
