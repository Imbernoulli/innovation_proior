import sys, random
seed=int(sys.argv[1]); random.seed(seed)
n=random.randint(1,40)
k=random.randint(1,n)
maxb=n//k
m=random.randint(1,max(1,maxb+1))
day_hi=random.randint(1,15)
b=[random.randint(1,day_hi) for _ in range(n)]
print(f"{n} {m} {k}")
print(" ".join(map(str,b)))
