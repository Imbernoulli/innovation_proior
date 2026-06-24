import sys, random

seed = int(sys.argv[1])
rng = random.Random(seed)

# small alphabet to force lots of overlapping occurrences (exercises the
# failure-chain propagation and double-count corners)
alpha_size = rng.choice([1, 2, 2, 2, 3])
alphabet = 'abc'[:alpha_size]

n = rng.randint(1, 12)
s = ''.join(rng.choice(alphabet) for _ in range(n))
print(s)
