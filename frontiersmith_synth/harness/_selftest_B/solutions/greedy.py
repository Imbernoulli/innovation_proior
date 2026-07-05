# TIER: greedy
import sys,json,math
inst=json.load(sys.stdin); pts=inst["points"]; n=len(pts)
vis=[False]*n; order=[0]; vis[0]=True; cur=0
for _ in range(n-1):
    nxt=min((i for i in range(n) if not vis[i]), key=lambda i: math.dist(pts[cur],pts[i]))
    vis[nxt]=True; order.append(nxt); cur=nxt
print(json.dumps({"tour":order}))
