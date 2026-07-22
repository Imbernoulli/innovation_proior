# TIER: greedy
# The obvious "read the data, sort by count" recipe: pin the pin_budget keys with the
# highest GLOBAL frequency, summed across all five traces, and keep plain LRU. This
# wins comfortably on zipf/phase/bursty (whose truly hot keys tend to also be globally
# frequent) -- but it is blind to the *worst*-trace objective: the adversarial-loop
# trace's cyclic period exceeds the shrunken managed region under plain LRU (the
# classic worst case for recency-only eviction), and the scan trace's one-shot sweep
# of fresh keys keeps flushing whatever the policy tried to protect. Since the score
# is the MINIMUM over all five traces, those two failures -- not the four wins --
# decide the grade.
import sys, json
from collections import Counter

inst = json.load(sys.stdin)
P = inst["pin_budget"]
traces = inst["traces"]

freq_all = Counter()
for tr in traces.values():
    freq_all.update(tr)

pin = [k for k, _ in freq_all.most_common(P)]

print(json.dumps({"pin": pin, "w_lru": 1.0, "w_mru": 0.0, "w_lfu": 0.0, "w_scan": 0.0}))
