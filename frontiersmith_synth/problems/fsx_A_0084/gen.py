import sys

# Glacier sensor net (Heilbronn-in-a-triangle variant).
# Difficulty ladder: number of sensors N grows small -> larger.
# Deterministic in testId only. The survey region is the fixed right
# triangle with corners (0,0), (1,0), (0,1); only N varies.
LADDER = [6, 7, 8, 9, 11, 12, 14, 16, 18, 21]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
N = LADDER[idx]

# instance: N, then the three triangle corners (counter-clockwise)
print(N)
print("0.0 0.0")
print("1.0 0.0")
print("0.0 1.0")
