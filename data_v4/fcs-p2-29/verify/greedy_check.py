# Tempting greedy: compute plain max-subarray (Kadane), then within THAT window delete its single most negative element.
import sys, itertools
def plain_kadane_window(a):
    n=len(a); best=None; bl=br=0; cur=0; cl=0
    for r in range(n):
        if cur<=0:
            cur=a[r]; cl=r
        else:
            cur+=a[r]
        if best is None or cur>best:
            best=cur; bl=cl; br=r
    return best, bl, br
def greedy(a):
    best,bl,br=plain_kadane_window(a)
    g=best
    window=a[bl:br+1]
    if len(window)>=2:
        g=max(g, best - min(window))
    return g
def brute(a):
    n=len(a); B=float('-inf')
    for l in range(n):
        s=0
        for r in range(l,n):
            s+=a[r]
            if s>B: B=s
            if r-l+1>=2:
                for k in range(l,r+1):
                    if s-a[k]>B: B=s-a[k]
    return B
# search small space for a counterexample
import random
rng=random.Random(1)
found=[]
for trial in range(200000):
    n=rng.randint(3,7)
    a=[rng.randint(-6,6) for _ in range(n)]
    if greedy(a)!=brute(a):
        found.append((a,greedy(a),brute(a)))
        if len(found)>=5: break
for f in found:
    print("a=",f[0]," greedy=",f[1]," correct=",f[2])
