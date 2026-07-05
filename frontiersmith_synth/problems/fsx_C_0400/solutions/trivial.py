# TIER: trivial
# Constant predictor: emit the MAJORITY label of the training logs for every test
# log.  This reproduces the evaluator's majority-of-train baseline exactly, so it
# scores ~0.1 on every instance -- it captures none of the length generalization.
import sys, json

inst = json.load(sys.stdin)
cnt = {0: 0, 1: 0, 2: 0}
for ex in inst["train"]:
    cnt[ex["label"]] += 1
best = 0
for lab in (0, 1, 2):
    if cnt[lab] > cnt[best]:
        best = lab

print(json.dumps({"labels": [best] * len(inst["test"])}))
