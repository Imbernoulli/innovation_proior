# TIER: trivial
# The naive strawman -- exactly the evaluator baseline. Steer the WHOLE beam to the
# single highest-weight patch with one blazed grating. Perfectly efficient into that
# one patch, but ignores the rest of the corridor -> reproduces the baseline -> ~0.1.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]
weights = inst["weights"]

k = int(np.argmax(np.array(weights, dtype=float)))
r, c = targets[k]
y = np.arange(N).reshape(N, 1)
x = np.arange(N).reshape(1, N)
fr = r - N // 2
fc = c - N // 2
phase = np.angle(np.exp(2j * np.pi * (fr * y + fc * x) / N))

print(json.dumps({"phase": phase.tolist()}))
