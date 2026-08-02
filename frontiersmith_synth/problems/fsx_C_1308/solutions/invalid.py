# TIER: invalid
# Dump every single item onto reviewer 0, ignoring their capacity entirely.
# Every instance in this family has reviewer capacities well below the queue
# length (that's the whole premise: too few reviewers), so this always
# overflows reviewer 0's capacity and the evaluator rejects it -> score 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

print(json.dumps({"schedule": {"0": list(range(n))}}))
