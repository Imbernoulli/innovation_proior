import sys

# Coral reef survey (Heilbronn-in-a-square variant of extremal-point-config).
# Difficulty ladder: number of survey stations N grows small -> larger.
# Deterministic in testId only. The reef plot is the fixed unit square with
# corners (0,0),(1,0),(1,1),(0,1); only N varies across tests.
LADDER = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
N = LADDER[idx]

# instance: N, then the four square corners (counter-clockwise)
print(N)
print("0.0 0.0")
print("1.0 0.0")
print("1.0 1.0")
print("0.0 1.0")
