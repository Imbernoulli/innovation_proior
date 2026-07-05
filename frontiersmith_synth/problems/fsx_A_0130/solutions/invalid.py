# TIER: invalid
# Send every car to track K (one past the last valid track index).  Every track
# is out of range, so the plan fails validation on every instance -> 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_cars"]
K = inst["n_tracks"]
print(json.dumps({"assign": [K] * N}))
