import sys

# Summit Ridge Resort (Heilbronn-in-the-unit-square variant).
# Difficulty ladder: number of lift towers N grows small -> larger, which
# raises the number of triples C(N,3) and makes thin slivers harder to avoid.
# Deterministic in testId only. The plot is the fixed unit square with corners
# (0,0), (1,0), (1,1), (0,1); only N varies.
LADDER = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
N = LADDER[idx]

print(N)
