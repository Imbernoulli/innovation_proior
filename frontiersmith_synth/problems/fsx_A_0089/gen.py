import sys

# Difficulty ladder: number of sensors grows small -> larger.
# Deterministic in testId only; dimension is fixed at 2 (unit square habitat floor).
LADDER = [6, 8, 10, 12, 16, 20, 25, 30, 36, 40]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
n = LADDER[idx]
d = 2
print("%d %d" % (n, d))
