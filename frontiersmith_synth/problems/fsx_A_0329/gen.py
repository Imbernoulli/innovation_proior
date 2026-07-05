import sys

# Difficulty ladder for a 3D formation-flying telescope constellation.
# Number of apertures grows small -> large; the configuration space is the
# unit cube [0,1]^3 (two ground-projected baseline axes + one delay/altitude axis).
# Deterministic in testId only; dimension fixed at 3.
LADDER = [8, 10, 12, 14, 16, 18, 20, 24, 27, 30]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
n = LADDER[idx]
d = 3
print("%d %d" % (n, d))
