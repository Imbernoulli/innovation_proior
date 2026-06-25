import random, sys
rng = random.Random(99)
n = 200000
print(n)
print(" ".join(str(rng.randint(-10**9, 10**9)) for _ in range(n)))
