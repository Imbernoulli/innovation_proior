# TIER: invalid
"""Broken routine: emits edges that reference an out-of-range subsystem index
(n_nodes, one past the last valid subsystem), so the evaluator rejects the answer
on every rig -> scores 0."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    d = int(inst["n_nodes"])
    # d is NOT a valid node index (valid indices are 0..d-1) -> parse rejects it
    print(json.dumps({"edges": [[0, d], [1, d]]}))


main()
