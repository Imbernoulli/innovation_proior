# TIER: invalid
"""Invalid answer: names intersections that do not exist (out-of-range indices).

The evaluator strictly validates that every edge references a real intersection in
[0, n_nodes); this answer violates that on every grid, so it is rejected and scores
exactly 0."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    p = int(inst["n_nodes"])
    # reference a non-existent intersection -> hard reject
    print(json.dumps({"edges": [[0, p + 50], [1, 2], [3, 4]]}))


main()
