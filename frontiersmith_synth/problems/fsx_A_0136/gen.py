import sys

# Difficulty ladder: number of buses N in the power-grid phase-pattern matrix.
# Deterministic in testId only. Mix of Hadamard sizes (4,8,16 -> attainable bound)
# and "gap" sizes (6,10,12,14,20,24,28 -> no exact Hadamard construction reachable
# by a submatrix, so the objective stays genuinely open-ended).
LADDER = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
N = LADDER[idx]
print(N)
