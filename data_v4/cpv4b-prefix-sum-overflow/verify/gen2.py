import sys, random
seed=int(sys.argv[1]); rng=random.Random(seed)
n=rng.randint(0,60)
mag=rng.choice([2,5,50,1000,1000000000])
a=[rng.randint(-mag,mag) for _ in range(n)]
m=rng.randint(0,3)
if m==0 and n>0:
    i=rng.randint(0,n-1); j=rng.randint(i,n-1); S=sum(a[i:j+1])
elif m==1: S=0
elif m==2: S=rng.randint(-3*mag,3*mag)
else: S=rng.randint(-200000000000000,200000000000000)
print(n,S); print(' '.join(map(str,a)))
