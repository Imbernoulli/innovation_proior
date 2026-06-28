boxes_in=[(6,6,10),(5,9,9),(4,8,8)]
oriented=[]
labels=[]
for idx,(x,y,z) in enumerate(boxes_in):
    d=[x,y,z]
    for k in range(3):
        h=d[k]; a=d[(k+1)%3]; b=d[(k+2)%3]
        if a>b: a,b=b,a
        oriented.append((a,b,h)); labels.append(f"box{idx}({x},{y},{z}) h={h} base=({a},{b})")
for o,l in zip(oriented,labels): print(l)
m=len(oriented)
import sys; sys.setrecursionlimit(10000)
memo={}; choice={}
def best(i):
    if i in memo: return memo[i]
    wi,di,hi=oriented[i]; r=hi; ch=None
    for j in range(m):
        wj,dj,hj=oriented[j]
        if wi>wj and di>dj:
            v=hi+best(j)
            if v>r: r=v; ch=j
    memo[i]=r; choice[i]=ch; return r
opt=0; start=None
for i in range(m):
    v=best(i)
    if v>opt: opt=v; start=i
print("OPTIMUM=",opt)
# reconstruct
cur=start; stack=[]
while cur is not None:
    stack.append((labels[cur], oriented[cur]))
    cur=choice[cur]
print("STACK (bottom->top):")
for s in stack: print("  ",s)
# verify what sol.cpp prints
