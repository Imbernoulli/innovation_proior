# TIER: strong
# Matched multi-transform policy.  The test-time jitter is translation + additive
# Gaussian noise + a global brightness offset, so the policy augments with exactly
# those three label-preserving transforms and spends most of the copy budget on the
# dominant nuisance (translation), with a lighter dose of noise and brightness:
#   shift(mag=3) x4  +  noise(std=0.15) x1  +  bright(delta=0.12) x1   (6 of 10 copies)
# A real solver would read train_images and estimate the shift spread / background
# noise to set these magnitudes; the fixed values below already track the family.
# It deliberately avoids mirror-flips / heavy contrast / cutout -- transforms the
# test distribution never contains, which would plant confusers and hurt 1-NN.
# It stays under the copy cap, so the oracle's far larger budget keeps it below 1.0.
import sys, json

inst = json.load(sys.stdin)
lim = inst.get("limits", {})
cap = int(lim.get("total_copies", 10))

ops = [
    {"type": "shift",  "mag": 3,     "copies": 4},
    {"type": "noise",  "std": 0.15,  "copies": 1},
    {"type": "bright", "delta": 0.12, "copies": 1},
]
# defensive: never exceed the advertised copy budget
while sum(o["copies"] for o in ops) > cap and ops[0]["copies"] > 1:
    ops[0]["copies"] -= 1

print(json.dumps({"ops": ops}))
