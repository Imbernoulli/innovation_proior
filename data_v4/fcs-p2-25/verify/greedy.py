import sys
# Greedy: start at the smallest cell in row 0, then always step to the smallest of the <=3 reachable cells.
def greedy(n, g):
    if n==0: return 0
    # choose smallest start in row 0
    col = min(range(n), key=lambda j: g[0][j])
    total = g[0][col]
    for i in range(1,n):
        cands = [c for c in (col-1,col,col+1) if 0<=c<n]
        col = min(cands, key=lambda c: g[i][c])
        total += g[i][col]
    return total
data=sys.stdin.read().split()
idx=0; n=int(data[idx]); idx+=1
g=[]
for i in range(n):
    r=[]
    for j in range(n):
        r.append(int(data[idx])); idx+=1
    g.append(r)
print(greedy(n,g))
