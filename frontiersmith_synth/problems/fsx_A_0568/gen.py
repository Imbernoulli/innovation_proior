import sys, random

# Calder-style hanging mobile atelier.
# Instance = a multiset of N integer leaf weights + an arm-length ceiling Amax.
# Difficulty ladder: N (a power of two) grows, and the weight multiset is planted
# with a rich divisor structure (highly-composite base) so that a clever grouping
# of leaves into equal-ish / high-gcd sibling subtrees unlocks large rod scaling,
# while the obvious "balance with minimal arms" approach cannot spread the silhouette.
#
# Deterministic in testId only.
LADDER_N = [4, 4, 8, 8, 16, 16, 16, 32, 32, 32]
BASE     = [6, 12, 6, 24, 12, 30, 24, 12, 60, 24]
MULT_HI  = [8, 8, 8, 10, 8, 8, 10, 8, 6, 10]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER_N)) - 1
N = LADDER_N[idx]
base = BASE[idx]
hi = MULT_HI[idx]

rng = random.Random(90000 + 137 * i)

# Planted structure: roughly half the leaves are "twins" (equal weights) so that a
# gcd-aware grouping can build equal-weight siblings -> rod scale can go all the way
# to Amax (wide, sculptable silhouette). The rest are varied multiples of the base.
ws = []
n_twin_pairs = N // 4
for _ in range(n_twin_pairs):
    w = base * rng.randint(1, hi)
    ws.append(w); ws.append(w)
while len(ws) < N:
    ws.append(base * rng.randint(1, hi))

rng.shuffle(ws)
Amax = sum(ws)  # ceiling generous enough that any balanced grouping is feasible

print("%d %d" % (N, Amax))
print(" ".join(str(w) for w in ws))
