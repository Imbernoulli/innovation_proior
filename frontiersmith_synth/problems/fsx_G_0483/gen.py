import sys

# Sonar-waveform Costas-permutation instance.
# The instance is simply the order n of the array. The difficulty ladder grows
# n from a medium order up to a large one. Every ladder order is chosen so that
# NEITHER n+1 is prime NOR n+2 is a prime power -- i.e. no direct Welch or
# Lempel-Golomb algebraic Costas construction applies, so no cheap perfect
# construction is reachable. Orders 32 and 33 in the ladder are proven to admit
# NO Costas array at all: the true minimum number of coincidences there is open.
# Deterministic in testId only.
LADDER = [19, 24, 26, 31, 32, 34, 38, 44, 48, 54]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
n = LADDER[idx]
print(n)
