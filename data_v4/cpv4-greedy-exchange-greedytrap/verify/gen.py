import random
import sys

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)

# Tiny cases to stress the greedy trap: small distinct denominations, small target.
n = random.randint(1, 5)
denoms = random.sample(range(1, 13), n)   # distinct denominations in [1,12]
A = random.randint(0, 30)

print(n, A)
print(" ".join(map(str, denoms)))
