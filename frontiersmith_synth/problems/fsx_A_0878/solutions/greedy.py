# TIER: greedy
# Blind forecast trust ("Belady-on-the-forecast"): always evict whichever
# resident pallet the runner's forecast claims is safest, no calibration, no
# phase awareness. This is the obvious first move once a "future forecast" is
# handed to you -- and it is a trap: it wins big when the runner is reliable,
# but on stretches where the runner is guessing (or actively backwards) it
# gets systematically misled into evicting pallets that come right back,
# scoring far below the always-LRU baseline on those instances.
import sys, json

json.load(sys.stdin)  # calibration data is available but unused
print(json.dumps({"mode": "hint"}))
