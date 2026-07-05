# TIER: trivial
# Majority-class constant classifier: every instance is built so that "not
# well-spliced" is the strict majority, so predicting 0 for every query recovers
# the majority-constant baseline the evaluator anchors to (-> ~0.1). No learning.
import sys, json

inst = json.load(sys.stdin)
q = inst["queries"]
print(json.dumps({"labels": [0] * len(q)}))
