# TIER: trivial
# Majority-class predictor: label every query paper with the most frequent seed
# subfield (ties broken toward the lowest class index).  This reproduces the
# evaluator's weak reference exactly, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
k = inst["k"]
train_labels = inst["train_labels"]
query_ids = inst["query_ids"]

counts = [0] * k
for c in train_labels:
    counts[c] += 1
maj = 0
for c in range(1, k):
    if counts[c] > counts[maj]:
        maj = c

print(json.dumps({"labels": [maj] * len(query_ids)}))
