import sys

# Difficulty ladder: number of telescope stations to place in the triangular
# reserve, growing from few (easy, fat triangles achievable) to many (hard,
# the minimum-area triple becomes very thin). Deterministic in testId only.
LADDER = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
N = LADDER[idx]
print("%d" % N)
