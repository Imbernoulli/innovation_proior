import sys, random

seed = int(sys.argv[1])
random.seed(seed)

# Small cases: keep k**n enumerable by the brute force (<= ~300000).
while True:
    n = random.randint(1, 12)
    k = random.randint(1, 7)
    if k ** n <= 300000:
        break

print(n, k)
