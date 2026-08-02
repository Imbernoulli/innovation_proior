import sys

# Difficulty ladder: number of mesh nodes n grows small -> large; clique size k fixed at 4
# so that R(4,4)=18 guarantees every test (n>=18) has at least one monochromatic K4 in ANY
# coloring (no degenerate zero-violation instance, no saturation). Deterministic in testId only.
LADDER = [18, 20, 24, 27, 30, 33, 36, 39, 42, 45]
K = 4

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
n = LADDER[idx]
print("%d %d" % (n, K))
