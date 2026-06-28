#!/usr/bin/env python3
"""
Trivial baseline: append requests in input order, keeping the route feasible.
Reads instance on stdin, writes a (feasible) solution on stdout. Used only to
establish the bar the heuristic solver must beat.
"""
import sys, math
L = 1000
def cd(ax,ay,bx,by):
    return int(math.ceil(math.hypot(ax-bx,ay-by)-1e-9))
def main():
    data=sys.stdin.read().split()
    it=iter(data)
    N=int(next(it)); T=int(next(it))
    req=[(L//2,L//2,0,0,0)]
    for _ in range(N):
        x=int(next(it)); y=int(next(it)); r=int(next(it)); d=int(next(it)); s=int(next(it))
        req.append((x,y,r,d,s))
    route=[]
    cx,cy=req[0][0],req[0][1]; t=0
    for i in range(1,N+1):
        x,y,r,d,s=req[i]
        arr=t+cd(cx,cy,x,y)
        start=max(arr,r)
        if start>d: continue
        after=start+s
        # can we still get home in time?
        if after+cd(x,y,req[0][0],req[0][1])>T: continue
        route.append(i); cx,cy=x,y; t=after
    out=[str(len(route))]
    if route: out.append(" ".join(map(str,route)))
    sys.stdout.write("\n".join(out)+"\n")
if __name__=="__main__":
    main()
