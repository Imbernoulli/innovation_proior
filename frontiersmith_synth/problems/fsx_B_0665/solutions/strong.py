# TIER: strong
# Predicted-wait rule: evict the resident line whose OWN observed inter-access
# gap predicts the FARTHEST next use. This is an online, per-line, self-estimated
# analog of Belady's offline-optimal "evict what's needed farthest in the future".
# It automatically behaves like MRU on long single scans (the just-touched line's
# predicted wait ~ its whole period, the largest around), like a period-aware
# LRU/LFU hybrid on overloaded short loops (short-gap lines are always "due soon"
# and get kept), and discounts one-off/noise addresses (unknown gap -> sentinel,
# evicted first) -- no hardcoded regime switch needed, one feature classifies and
# adapts by itself.
import sys, json

json.load(sys.stdin)
print(json.dumps({"w0": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 1.0, "w5": 0.0}))
