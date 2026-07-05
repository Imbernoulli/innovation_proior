# TIER: invalid
# A broken solver: it computes a phase field but forgets to keep it 2-D, emitting a
# FLAT list of N*N values instead of the required N-by-N nested grid. The evaluator
# validates shape strictly (list of N rows, each of length N), so the answer is
# rejected as malformed and scores 0 on every instance -- exercising the feasibility
# gate.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
N = inst["N"]
phase = np.zeros(N * N)             # wrong shape: 1-D length N*N, not N-by-N
print(json.dumps({"phase": phase.tolist()}))
