import sys
from functools import lru_cache
def main():
    d=sys.stdin.read().split(); idx=0
    n=int(d[idx]);idx+=1; W=int(d[idx]);idx+=1
    wt=[];val=[]
    for i in range(n):
        wt.append(int(d[idx]));idx+=1; val.append(int(d[idx]));idx+=1
    sys.setrecursionlimit(10000)
    @lru_cache(maxsize=None)
    def rec(i,cap):
        if i==n: return 0
        best=rec(i+1,cap)
        if wt[i]<=cap:
            best=max(best, val[i]+rec(i+1,cap-wt[i]))
        return best
    print(rec(0,W))
main()
