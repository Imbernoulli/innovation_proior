# TIER: greedy
# The obvious first idea: read off every species' raw uptake rate (vmax) and fill
# the ENTIRE row with the single most efficient consumer. This ignores whatever
# it excretes and whether anything nearby could use that waste -- exactly the
# trap the family is built around: the fastest species is also usually the most
# self-inhibited, and monoculture means nobody relieves it.
import sys, json

inst = json.load(sys.stdin)
L = inst["L"]
species = inst["species"]

best_idx = max(range(len(species)), key=lambda i: species[i]["vmax"])
assign = [best_idx] * L
print(json.dumps({"assign": assign}))
