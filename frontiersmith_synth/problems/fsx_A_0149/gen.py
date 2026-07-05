import sys

# Geothermal well siting -- difficulty ladder: number of wells N grows small -> larger.
# Deterministic in testId only. The reservoir is the unit square [0,1]^2 (D = 2).
LADDER = [8, 12, 16, 21, 25, 30, 36, 42, 49, 56]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
N = LADDER[idx]
D = 2
print("%d %d" % (N, D))
