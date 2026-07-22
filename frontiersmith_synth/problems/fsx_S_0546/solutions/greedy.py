# TIER: greedy
import sys
PAIR=[[1,0,3,2],[3,2,1,0]]
def eid(m,r,c,d):
    if d==1: return r*m+c
    if d==3: return r*m+((c-1)%m)
    if d==2: return m*m+r*m+c
    return m*m+((r-1)%m)*m+c
def comps(m,tile):
    E=2*m*m; nbr=[[-1,-1] for _ in range(E)]; slot=[0]*E
    for r in range(m):
        for c in range(m):
            t=tile[r*m+c]
            for a,b in ((0,PAIR[t][0]),(2,PAIR[t][2])):
                e1=eid(m,r,c,a); e2=eid(m,r,c,b)
                nbr[e1][slot[e1]]=e2; slot[e1]+=1
                nbr[e2][slot[e2]]=e1; slot[e2]+=1
    comp=[-1]*E; L=0
    for s in range(E):
        if comp[s]!=-1: continue
        prev=-1;cur=s
        while comp[cur]==-1:
            comp[cur]=L; a,b=nbr[cur]; nxt=a if a!=prev else b; prev=cur; cur=nxt
        L+=1
    return L,comp
def bridges(m,X):
    L,comp=comps(m,X)
    if L==1: return 1,[]
    out=[]
    for v in range(m*m):
        r=v//m; c=v%m
        if comp[eid(m,r,c,0)]!=comp[eid(m,r,c,2)]: out.append(v)
    return L,out
def readin():
    tok=sys.stdin.read().split(); it=iter(tok)
    m=int(next(it)); K=int(next(it)); lam=int(next(it))
    P=[int(next(it)) for _ in range(m*m)]
    w=[int(next(it)) for _ in range(m*m)]
    oid=[int(next(it)) for _ in range(m*m)]
    return m,K,lam,P,w,oid
def emit(m,X):
    out=[" ".join(str(X[r*m+c]) for c in range(m)) for r in range(m)]
    sys.stdout.write("\n".join(out)+"\n")
def main():
    m,K,lam,P,w,oid=readin()
    X=P[:]
    while True:
        L,br=bridges(m,X)
        if L==1 or not br: break
        X[min(br,key=lambda v:w[v])]^=1
    emit(m,X)
main()
