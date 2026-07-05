# TIER: trivial
# In-phase superposition of single-spot gratings, keep only the argument.
# This is EXACTLY the evaluator's reference construction -> normalizes to ~0.1.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
targets = inst["targets"]
y = np.arange(N).reshape(-1, 1)
x = np.arange(N).reshape(1, -1)
C = np.zeros((N, N), dtype=complex)
for fy, fx in targets:
    C += np.exp(2j * np.pi * (fy * y + fx * x) / N)
phi = np.angle(C)
print(json.dumps({"phase": phi.tolist()}))
