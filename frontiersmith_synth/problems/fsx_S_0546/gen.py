import sys, random
# Toroidal single-stroke kolam mandala instance generator.
# testId 1..10 -> difficulty ladder (grid side m, all even for clean C4 orbits).
PAIR=[[1,0,3,2],[3,2,1,0]]
def eid(m,r,c,d):
    if d==1: return r*m+c
    if d==3: return r*m+((c-1)%m)
    if d==2: return m*m+r*m+c
    return m*m+((r-1)%m)*m+c
def loops(m,tile):
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
    return L
def rot(m,r,c): return (c,m-1-r)
def build_sym(m,seed):
    # C4-symmetric preferred pattern P and orbit ids; tile flips under 90-deg rotation.
    rng=random.Random(seed); P=[-1]*(m*m); oid=[-1]*(m*m); k=0
    for r in range(m):
        for c in range(m):
            v=r*m+c
            if P[v]!=-1: continue
            t=rng.randint(0,1); cr,cc=r,c; cur=t
            for _ in range(4):
                P[cr*m+cc]=cur; oid[cr*m+cc]=k; cr,cc=rot(m,cr,cc); cur=1-cur
            k+=1
    return P,oid

def main():
    tid=int(sys.argv[1])
    ladder=[6,8,8,10,10,12,12,14,14,16]
    m=ladder[(tid-1)%len(ladder)]
    K=4; LAM=80
    # seed the symmetric target; guarantee it is INFEASIBLE (>=2 loops: the trap).
    s=0
    while True:
        P,oid=build_sym(m,20250700+tid*97+m*7+s)
        if loops(m,P)>=2: break
        s+=1
    wrng=random.Random(30000+tid*131+m)
    w=[wrng.randint(20,120) for _ in range(m*m)]
    out=[]
    out.append("%d %d %d"%(m,K,LAM))
    for r in range(m): out.append(" ".join(str(P[r*m+c]) for c in range(m)))
    for r in range(m): out.append(" ".join(str(w[r*m+c]) for c in range(m)))
    for r in range(m): out.append(" ".join(str(oid[r*m+c]) for c in range(m)))
    sys.stdout.write("\n".join(out)+"\n")
main()
