# TIER: invalid
# Emit a plan that blows the migration budget and ignores the power-of-two candidate
# restriction: move EVERY key to shard 0 regardless of its allowed {shard0, alt0, alt1}
# set. The evaluator rejects any assign[i] outside the allowed set and any move-count
# above budget, so this scores 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"assign": [0] * N}))
