import sys,random
i=int(sys.argv[1]); rng=random.Random(1000+i)
n=3 if i<=1 else i*40
print(n)
for _ in range(n): print(rng.randint(1,100), rng.randint(1,100))
