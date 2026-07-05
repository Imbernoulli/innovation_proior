import sys

# Tide Pool Biodiversity Survey (format C, minimize 3D star discrepancy).
# Difficulty ladder: number of survey quadrats grows small -> larger.
# Deterministic in testId only; the environmental cube is always [0,1]^3
# with axes (tidal elevation, salinity, substrate rugosity), so d = 3.
LADDER = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
n = LADDER[idx]
d = 3
print("%d %d" % (n, d))
