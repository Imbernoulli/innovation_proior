# TIER: strong
# Best-fit priority, tuned per instance. The core rule minimizes leftover budget
# after stowing (phi2 = (res-s)/C, weight -1), which closes modules densely and is
# the strongest simple online rule under the fixed streaming simulator. We also add
# a small negative weight on the squared-leftover term (phi3), which reinforces the
# preference for near-exact fits on skewed cost distributions (bimodal / heavy /
# Weibull) where many small canisters must top off partially-filled modules. The
# ordering stays monotone in leftover, so this behaves as best-fit while nudging
# tie-adjacent choices toward tighter packs. Because the L1 bound is loose, even
# this stays well below 1.0 on most instances -> headroom remains.
import sys, json

inst = json.load(sys.stdin)
items = inst.get("items", [])
C = inst.get("capacity", 1)

# light instance-aware tuning: if the stream is dominated by small canisters, lean
# harder on the squared-leftover term to favor exact tops-off.
frac_small = (sum(1 for s in items if s <= max(1, C // 4)) / len(items)) if items else 0.0
w3 = -0.6 if frac_small >= 0.4 else -0.3

print(json.dumps({"weights": [0.0, 0.0, -1.0, w3]}))
