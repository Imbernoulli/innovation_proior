# TIER: trivial
# Majority-label rule: a one-state DFA that predicts the train-majority label on every
# trellis, ignoring structure entirely.  This reproduces the evaluator's majority
# baseline, so it scores ~0.1 on every instance (and shows zero ID->OOD generalization).
import sys, json

inst = json.load(sys.stdin)
D = inst["n_types"]
train = inst["train"]
ones = sum(1 for _, y in train if y == 1)
maj = 1 if ones * 2 >= len(train) else 0

# one state, self-loop on every symbol, accepting iff maj==1
print(json.dumps({"start": 0, "accept": [maj], "trans": [[0] * D]}))
