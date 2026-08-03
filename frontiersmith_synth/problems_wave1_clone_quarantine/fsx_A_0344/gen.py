import sys

# Cryo Qubit Junction Layout (Heilbronn-in-the-unit-TRIANGLE variant).
# A triangular cryostat plate carries N wire-bond junction pads. Difficulty
# ladder: number of pads N grows small -> larger, which raises the number of
# triples C(N,3) and makes thin sliver triangles harder to avoid.
# Deterministic in testId only. The plate is the fixed unit right triangle with
# corners (0,0), (1,0), (0,1); only N varies across tests.
LADDER = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
N = LADDER[idx]

print(N)
