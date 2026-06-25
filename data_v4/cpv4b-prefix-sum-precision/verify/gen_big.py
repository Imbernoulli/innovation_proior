import sys, random
seed=int(sys.argv[1]); rng=random.Random(seed)
n=rng.randint(2000,4000); L=rng.randint(1,n)
mode=rng.randint(0,2)
vals=[]
for _ in range(n):
    if mode==0: vals.append(rng.randint(-10**9,10**9))
    elif mode==1: vals.append(rng.choice([-10**9,10**9]))
    else: vals.append(rng.randint(-10**9,10**9))
print(n,L); print(' '.join(map(str,vals)))
