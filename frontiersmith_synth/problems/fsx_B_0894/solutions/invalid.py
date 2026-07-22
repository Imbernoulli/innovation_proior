# TIER: invalid
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]
P = inst["budget"]

# Deliberately broken: dumps ALL of the power budget onto every single line
# at once (grossly infeasible -- sum(power) far exceeds the budget) and
# repeats one duplicate index in "order" instead of a permutation.
power = [P] * N
order = [0] * N

print(json.dumps({"power": power, "order": order}))
