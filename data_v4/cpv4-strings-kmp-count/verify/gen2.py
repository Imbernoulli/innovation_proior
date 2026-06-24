import sys, random
seed=int(sys.argv[1]); rng=random.Random(seed)
a=rng.choice([1,2,2,3]); alpha='abcd'[:a]
n=rng.randint(1,40)
print(''.join(rng.choice(alpha) for _ in range(n)))
