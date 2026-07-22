# TIER: trivial
# Ignore the strings entirely. Emit a single-state DFA that always predicts the
# majority training label (self-loop on both symbols). Reproduces the evaluator's
# weak "a_triv" anchor exactly -> normalizes to ~0.1 on every device.
import sys, json

inst = json.load(sys.stdin)
train = inst["train"]
pos = sum(1 for t in train if t["label"] == 1)
neg = len(train) - pos
maj = 1 if pos >= neg else 0

delta = [[0, 0]]
accept = [0] if maj == 1 else []
print(json.dumps({"delta": delta, "start": 0, "accept": accept}))
