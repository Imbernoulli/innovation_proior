import sys, random
# wider-magnitude small-case generator (still tiny n for the exponential brute)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = random.Random(seed)
n = rng.randint(0, 7)
vals = [rng.randint(-9, 9) for _ in range(n)]
print(n)
if n:
    print(" ".join(map(str, vals)))
