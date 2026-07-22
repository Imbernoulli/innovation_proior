# TIER: greedy
# Pure LRU: evict the resident line with the largest "time since last use".
# The obvious first cache policy anyone reaches for -- fine on short loops that
# fit the cache, but collapses to near-zero hit rate on any cyclic scan whose
# period exceeds cache_size (sequential-flooding pathology).
import sys, json

json.load(sys.stdin)
print(json.dumps({"w0": 0.0, "w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.0, "w5": 0.0}))
