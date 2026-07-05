# TIER: trivial
# Emit NO edges at all -- the empty influence map.  This is the calibration
# baseline: its SHD equals the number of true edges, so it scores ~0.1.
import sys, json
inst = json.load(sys.stdin)
print(json.dumps({"edges": []}))
