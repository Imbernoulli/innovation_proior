# TIER: greedy
# Marginal-frequency order: place the assay stage required by the most probes
# first, so its calibration is shared (a cache hit) by the largest number of
# probes, then the next-most-frequent, and so on. Ties broken by stage id for
# determinism. Captures popularity but ignores stage-to-stage correlation.
import sys, json
inst = json.load(sys.stdin)
C = inst["n_stages"]
probes = inst["probes"]
freq = [0] * C
for S in probes:
    for c in S:
        freq[c] += 1
order = sorted(range(C), key=lambda c: (-freq[c], c))
print(json.dumps({"order": order}))
