# TIER: trivial
# The cheapest fabricable mask -- a BINARY (two-level, 0/pi) phase mask, made by
# thresholding the grating superposition. This is exactly the evaluator's baseline
# construction, so it reproduces b and scores ~0.1 on every instance. Binary etching
# is cheap but throws away half the control (conjugate ghost orders steal power and
# the fan comes out uneven), so it leaves lots of room for better analog designs.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]

y = np.arange(N).reshape(N, 1)
x = np.arange(N).reshape(1, N)
acc = np.zeros((N, N), dtype=complex)
for (r, c) in targets:
    acc += np.exp(2j * np.pi * ((r - N // 2) * y + (c - N // 2) * x) / N)
phase = np.where(acc.real >= 0.0, 0.0, np.pi)

print(json.dumps({"phase": phase.tolist()}))
