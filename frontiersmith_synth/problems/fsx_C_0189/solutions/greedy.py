# TIER: greedy
# Textbook constant-step extragradient: eta_k = gamma_k = 1/L, where L = ||M||_2 is the
# operator's Lipschitz constant. Stable and monotone, but a single fixed step wastes the
# finite budget -- no acceleration, no per-step tuning.
import sys, json
import numpy as np
inst = json.load(sys.stdin)
M = np.array(inst["M"], dtype=float)
K = inst["K"]
L = float(np.linalg.norm(M, 2))
step = 1.0 / max(L, 1e-9)
print(json.dumps({"eta": [step] * K, "gamma": [step] * K}))
