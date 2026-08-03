import sys

# Difficulty ladder: number of sweeper drones grows small -> larger.
# Deterministic in testId only; the sector is the unit square.
LADDER = [3, 5, 7, 10, 13, 17, 21, 26, 33, 40]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
N = LADDER[idx]
S = 1.0
print("%d %.1f" % (N, S))
