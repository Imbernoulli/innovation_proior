import sys

# Pandemic contact net (Heilbronn-in-the-unit-square variant).
# Difficulty ladder: number of attendees N grows small -> large.
# Deterministic in testId only. The hall is the fixed unit square with
# corners (0,0),(1,0),(1,1),(0,1); only N varies across tests.
LADDER = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
N = LADDER[idx]

# instance: N, then the four square corners (counter-clockwise)
print(N)
print("0.0 0.0")
print("1.0 0.0")
print("1.0 1.0")
print("0.0 1.0")
