g=[[5,5,5],[1,9,9],[50,50,1]]
n=3
best=None; bestpath=None
def dfs(r,c,acc,path):
    global best,bestpath
    acc+=g[r][c]; path=path+[c]
    if r==n-1:
        if best is None or acc<best:
            best=acc; bestpath=path[:]
        return
    for dc in(-1,0,1):
        nc=c+dc
        if 0<=nc<n: dfs(r+1,nc,acc,path)
for s in range(n): dfs(0,s,0,[])
print("optimal sum",best,"path cols",bestpath)
# greedy starting at col0
col=0; tot=g[0][0]; path=[0]
for i in range(1,n):
    cands=[c for c in (col-1,col,col+1) if 0<=c<n]
    col=min(cands,key=lambda c:g[i][c]); tot+=g[i][col]; path.append(col)
print("greedy(start col0) sum",tot,"path cols",path)
