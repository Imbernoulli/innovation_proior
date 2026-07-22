import sys
# Deterministic scorer for the toroidal single-stroke kolam mandala.
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
def costfn(P,X,w,oid,K,lam):
    oc=[0]*K; s=0
    for i in range(len(P)):
        if X[i]!=P[i]:
            s+=w[i]; oc[oid[i]]+=1
    return s+lam*sum(d*(d-1)//2 for d in oc)
def merge_first(m,P):
    X=P[:]
    while True:
        L,br=bridges(m,X)
        if L==1 or not br: break
        X[br[0]]^=1
    return X
def merge_marg(m,P,w,oid,K,lam,seed=None,topk=1):
    import random as _r
    rng=_r.Random(seed) if seed is not None else None
    X=P[:]
    while True:
        L,br=bridges(m,X)
        if L==1 or not br: break
        Dcnt=[0]*K
        for i in range(len(X)):
            if X[i]!=P[i]: Dcnt[oid[i]]+=1
        scored=[]
        for v in br:
            was=1 if X[v]!=P[v] else 0
            d=Dcnt[oid[v]]
            marg=(w[v]+lam*d) if was==0 else (-w[v]-lam*(d-1))
            scored.append((marg,v))
        scored.sort()
        pick=scored[rng.randrange(min(topk,len(scored)))][1] if rng is not None else scored[0][1]
        X[pick]^=1
    return X
def checker_ref(m,P,w,oid,K,lam):
    best=merge_marg(m,P,w,oid,K,lam)
    bc=costfn(P,best,w,oid,K,lam)
    for s in range(8):
        c=merge_marg(m,P,w,oid,K,lam,seed=4242+s,topk=3)
        L,_=comps(m,c)
        if L==1:
            cc=costfn(P,c,w,oid,K,lam)
            if cc<bc: bc=cc
    return bc

def fail(msg):
    print("Ratio: 0.0 (%s)"%msg); sys.exit(0)

def main():
    intoks=open(sys.argv[1]).read().split()
    it=iter(intoks)
    try:
        m=int(next(it)); K=int(next(it)); lam=int(next(it))
        P=[int(next(it)) for _ in range(m*m)]
        w=[int(next(it)) for _ in range(m*m)]
        oid=[int(next(it)) for _ in range(m*m)]
    except Exception:
        fail("bad input")
    norb=max(oid)+1
    # participant output
    raw=open(sys.argv[2]).read().split()
    if len(raw)<m*m:
        fail("too few tokens")
    X=[]
    for t in raw[:m*m]:
        try:
            v=int(t)
        except Exception:
            fail("non-integer token")
        if v not in (0,1):
            fail("token not in {0,1}")
        X.append(v)
    # feasibility: exactly one loop covering all 2m^2 edges
    L,_=comps(m,X)
    if L!=1:
        fail("not single loop: %d loops"%L)
    cost=costfn(P,X,w,oid,norb,lam)
    cWorst=costfn(P,merge_first(m,P),w,oid,norb,lam)
    cBest=checker_ref(m,P,w,oid,norb,lam)
    denom=cWorst-cBest
    if denom<1.0: denom=1.0
    ratio=0.1+0.75*(cWorst-cost)/denom
    if ratio<0.0: ratio=0.0
    if ratio>1.0: ratio=1.0
    print("cost=%d cWorst=%d cBest=%d Ratio: %.6f"%(cost,cWorst,cBest,ratio))
main()
