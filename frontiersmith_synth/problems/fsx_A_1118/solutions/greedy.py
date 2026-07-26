# TIER: greedy
# Textbook online-knapsack recipe: estimate ONE static value-density cutoff from
# the preview manifest (its mean density) and commit to that constant for the
# whole tide -- no reaction to remaining capacity, no reaction to how the
# observed density level drifts once real lots start arriving.  This is the
# "obvious first idea" a solver reaches for, and it has no way to notice mid
# -stream that today diverged from what the preview implied.
import sys, json

inst = json.load(sys.stdin)
preview = inst["preview"]

densities = [value / size for size, value in preview]
mean_density = sum(densities) / len(densities)

policy = {"base": mean_density, "cap_gain": 0.0, "drift_gain": 0.0, "time_gain": 0.0}
print(json.dumps({"policy": policy}))
