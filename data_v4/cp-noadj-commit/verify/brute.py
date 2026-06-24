import sys
def solve(a):
    n=len(a); best=0
    for mask in range(1<<n):
        ok=True; s=0; prev=-2
        for i in range(n):
            if mask>>i&1:
                if i-prev==1: ok=False;break
                s+=a[i]; prev=i
        if ok: best=max(best,s)
    return best
data=sys.stdin.read().split()
n=int(data[0]); a=list(map(int,data[1:1+n]))
print(solve(a))
