# TIER: trivial
# Pure echo / memorizer: predict output == input for every test string.
# This is the classic length-generalization FAILURE mode -- it only ever gets the
# identity ("copy") instances right and scores ~0 on every real transformation,
# landing at the calibrated ~0.1 baseline.
import sys, json

inst = json.load(sys.stdin)
tests = inst["tests"]
print(json.dumps({"pred": list(tests)}))
