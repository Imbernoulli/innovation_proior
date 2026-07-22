# TIER: greedy
# The obvious approach: count every token in the observed prefix and keep the K
# most frequent so far. Equal-weight history -> it commits its whole budget to
# the pre-drift LEADERS (huge early counts) and the early DECOY spike, and never
# retains the late, still-small TRUE risers that surge after the window closes.
import sys, json
from collections import Counter
inst = json.load(sys.stdin)
K = inst["K"]; stream = inst["stream"]
c = Counter(stream)
keep = [t for t, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:K]]
print(json.dumps({"keep": keep}))
