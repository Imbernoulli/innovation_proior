import sys, random
seed=int(sys.argv[1]); rng=random.Random(seed)
n=rng.choice([150000,180000,200000]); L=rng.randint(1,n)
mode=rng.randint(0,1)
out=[]
for _ in range(n):
    if mode==0: out.append(str(rng.randint(-10**9,10**9)))
    else: out.append(str(rng.choice([-10**9,10**9])))
sys.stdout.write(f"{n} {L}\n"); sys.stdout.write(' '.join(out)+"\n")
