# TIER: greedy
# Analog grating-superposition mask: sum one unit blazed grating per requested spot
# and keep the FULL continuous phase (not just its sign). Using all phase levels
# instead of a two-level etch recovers a lot of efficiency and much better uniformity
# than the binary baseline in a single shot -- no iteration. But the phase-only
# projection still wastes power in stray orders and lights the spots unevenly, so it
# leaves headroom for iterative phase retrieval.
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
phase = np.angle(acc)

print(json.dumps({"phase": phase.tolist()}))
