# TIER: trivial
# No augmentation at all: hand back an empty policy.  The trainer builds the 1-NN
# gallery from the raw handful of sprites, reproducing the evaluator's weak
# no-augmentation reference, so this scores exactly ~0.1 on every instance.
import sys, json

json.load(sys.stdin)
print(json.dumps({"ops": []}))
