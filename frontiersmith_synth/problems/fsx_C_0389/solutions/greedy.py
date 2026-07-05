# TIER: greedy
# The obvious modern default: a single ReLU.  Rectified random features are a
# universal-ish basis, so this jumps well above the linear baseline on every
# inverter setting with zero design effort.  But one rectifier is one shape: it
# under-serves the even / localized responses (efficiency bell, MPP-tracking
# parabola, clip magnitude) and leaves accuracy on the table that a mixed shape
# recovers -- so it lands clearly below the strong tier.
import sys, json

json.load(sys.stdin)
print(json.dumps({"components": [{"base": "relu", "a": 1.0, "b": 0.0, "w": 1.0}]}))
