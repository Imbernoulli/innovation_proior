# TIER: strong
import sys,json,math
inst=json.load(sys.stdin); pts=inst["points"]; n=len(pts)
vis=[False]*n; order=[0]; vis[0]=True; cur=0
for _ in range(n-1):
    nxt=min((i for i in range(n) if not vis[i]), key=lambda i: math.dist(pts[cur],pts[i]))
    vis[nxt]=True; order.append(nxt); cur=nxt
def d(a,b): return math.dist(pts[a],pts[b])
imp=True
while imp:
    imp=False
    for i in range(n-1):
        for j in range(i+2,n):
            if i==0 and j==n-1: continue
            a,b,c,e=order[i],order[i+1],order[j],order[(j+1)%n]
            if d(a,b)+d(c,e) > d(a,c)+d(b,e)+1e-12:
                order[i+1:j+1]=order[i+1:j+1][::-1]; imp=True
print(json.dumps({"tour":order}))
