# TIER: invalid
"""Broken routine: emits edges that reference an out-of-range gallery index
(n_galleries, one past the last valid gallery), so the evaluator rejects the
answer on every museum -> scores 0."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    d = int(inst["n_galleries"])
    # d is NOT a valid gallery index (valid indices are 0..d-1) -> parse rejects it
    print(json.dumps({"edges": [[0, d], [1, d]]}))


main()
