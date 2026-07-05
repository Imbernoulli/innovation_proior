import sys
inp=open(sys.argv[1]).read().split(); out=open(sys.argv[2]).read().split()
it=iter(inp); n=int(next(it)); p=[];c=[];B=0
for _ in range(n):
    pi=int(next(it)); ci=int(next(it)); p.append(pi); c.append(ci); B+=pi
try: k=int(out[0]); ids=list(map(int,out[1:1+k]))
except Exception: print("Ratio: 0.0 (parse)"); sys.exit(0)
if len(ids)!=k: print("Ratio: 0.0 (count)"); sys.exit(0)
cov=[False]*n; seen=set()
for idx in ids:
    if idx<1 or idx>n or idx in seen: print("Ratio: 0.0 (bad idx %d)"%idx); sys.exit(0)
    seen.add(idx); cov[idx-1]=True
F=sum(c[i] if cov[i] else p[i] for i in range(n))
sc=min(1000.0,100.0*B/max(1,F))
print("F=%d B=%d Ratio: %.6f"%(F,B,sc/1000.0))
