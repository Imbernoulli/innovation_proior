import sys,random
r=random.Random(int(sys.argv[1]))
n=r.randint(0,10)
C=r.randint(0,6)
kr=r.random()
if kr<0.2: K=0
elif kr<0.35: K=r.randint(n+1,n+2)
else: K=r.randint(0,n) if n>0 else 0
print(n,K,C)
for _ in range(n):
    wr=r.random()
    if wr<0.35: w=0
    elif wr<0.5: w=r.randint(C+1,C+3)
    else: w=r.randint(0,max(0,C))
    print(w, r.randint(-5,5))
