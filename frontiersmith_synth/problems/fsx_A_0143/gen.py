import sys

# Difficulty ladder: number of control zones grows small -> larger.
# Deterministic in testId only. The arena is a disk of radius R = 1 centred at the origin.
LADDER = [3, 5, 7, 10, 13, 17, 21, 26, 33, 40]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
N = LADDER[idx]
R = 1.0
print("%d %.1f" % (N, R))
