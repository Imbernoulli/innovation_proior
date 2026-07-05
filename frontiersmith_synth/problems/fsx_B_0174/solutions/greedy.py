# TIER: greedy
# Frequency-descending order: put the most-used sensor modules first so many
# tool strings share their leading calibration. Ignores co-occurrence structure
# (tightly-grouped modules get interleaved with independent noise), so it beats
# the naive reference but leaves reuse on the table.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
wells = inst["wells"]
cnt = [0] * K
for w in wells:
    for m in w:
        cnt[m] += 1
order = sorted(range(K), key=lambda m: (-cnt[m], m))
print(json.dumps({"order": order}))
