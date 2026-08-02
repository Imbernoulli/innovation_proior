# TIER: trivial
# Never reinvest: harvest 100% of every turn's output. Capital never grows,
# no threshold is ever crossed. Reproduces the evaluator's weak baseline
# exactly, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_turns"]

print(json.dumps({"invest": [0.0] * N}))
