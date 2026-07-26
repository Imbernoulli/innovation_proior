# TIER: invalid
# Quotes every job a delay of 0. Since every job's service time is >= 2, this
# always violates "delay >= service", so the evaluator rejects the whole answer
# for every instance -> scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

decisions = [{"action": "quote", "delay": 0} for _ in range(n)]
print(json.dumps({"decisions": decisions}))
