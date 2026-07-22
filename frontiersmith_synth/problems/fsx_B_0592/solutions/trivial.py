# TIER: trivial
# Do-nothing baseline: keep K distinct tokens chosen uniformly at random
# (seeded deterministically from the public input). No counting, no drift model.
import sys, json, random
inst = json.load(sys.stdin)
K = inst["K"]; stream = inst["stream"]
distinct = sorted(set(stream))
rng = random.Random(1000003 * len(stream) + 31 * K + inst["M"])
rng.shuffle(distinct)
print(json.dumps({"keep": distinct[:K]}))
