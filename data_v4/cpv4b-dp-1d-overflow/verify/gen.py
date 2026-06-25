import sys, random

# Random SMALL-case generator for sol-vs-brute checking.
# Brute enumerates (m+1)^n sequences, so keep n and m tiny.
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)

n = random.randint(0, 7)
m = random.randint(0, 4)
print(n, m)
