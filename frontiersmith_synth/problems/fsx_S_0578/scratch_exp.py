import sys

def bubble_net(n):
    c=[]
    for i in range(n):
        for j in range(n-1):
            c.append((j,j+1))  # full OETS-like? no, this is bubble passes
    return c

def bubble_insertion(n):
    # standard insertion/bubble sorting network: n(n-1)/2 comparators
    c=[]
    for i in range(1,n):
        for j in range(i,0,-1):
            c.append((j-1,j))
    return c

def oddeven_transposition_round(n, parity):
    c=[]
    start = 0 if parity==0 else 1
    for i in range(start, n-1, 2):
        c.append((i,i+1))
    return c

def oets_rounds(n, R):
    c=[]
    for r in range(R):
        c += oddeven_transposition_round(n, r%2)
    return c

def batcher(n):
    # odd-even mergesort built on next-pow2, filtered to indices < n
    import math
    m=1
    while m<n: m*=2
    comps=[]
    def merge(lo, nn, r):
        step=r*2
        if step<nn:
            merge(lo,nn,step)
            merge(lo+r,nn,step)
            i=lo+r
            while i+r < lo+nn:
                comps.append((i,i+r))
                i+=step
        else:
            comps.append((lo,lo+r))
    def sort(lo,nn):
        if nn>1:
            mm=nn//2
            sort(lo,mm)
            sort(lo+mm,mm)
            merge(lo,nn,1)
    sort(0,m)
    # filter comparators touching padding wires >= n
    return [(a,b) for (a,b) in comps if a<n and b<n]

def apply_net_binary_all(n, net):
    # bit-parallel over all 2^n inputs. wire w -> integer where bit k = value of wire w for input k
    N=1<<n
    full=(1<<N)-1
    wires=[]
    for w in range(n):
        val=0
        # bit k of input = (k>>w)&1 ; wire w initial value for input k
        # build integer: for each k set bit k if (k>>w)&1
        # pattern: bit w of k. That's a repeating pattern.
        block=1<<w
        # construct
        v=0
        for k in range(N):
            if (k>>w)&1:
                v|=(1<<k)
        wires.append(v)
    for (i,j) in net:
        a=wires[i]; b=wires[j]
        wires[i]=a&b
        wires[j]=a|b
    return wires

def sorted_targets(n):
    N=1<<n
    tw=[]
    for w in range(n):
        v=0
        for k in range(N):
            p=bin(k).count("1")
            # ascending: wires [n-p .. n-1] are 1
            if w >= n-p:
                v|=(1<<k)
        tw.append(v)
    return tw

def sorts_all(n, net, targets):
    w=apply_net_binary_all(n,net)
    return w==targets

def tolerant(n, net, targets):
    # original must sort
    if not sorts_all(n,net,targets): return False
    for d in range(len(net)):
        sub=net[:d]+net[d+1:]
        if not sorts_all(n,sub,targets):
            return False
    return True

import time
for n in [7,8,9,10,11,12,13,14,15,16]:
    targets=sorted_targets(n)
    b=batcher(n); bub=bubble_insertion(n)
    assert sorts_all(n,b,targets), "batcher broken"
    assert sorts_all(n,bub,targets), "bubble broken"
    # find min R for batcher + R OETS rounds tolerant
    minR=None
    for R in range(0, n+3):
        net=b+oets_rounds(n,R)
        if tolerant(n,net,targets):
            minR=R; break
    tail = len(oets_rounds(n,minR)) if minR is not None else None
    strong_ct = len(b)+ (tail if tail else 0)
    B=2*len(bub); g=2*len(b); s=strong_ct
    def r(F):
        import math
        return min(1.0,0.1*(B/F)**2)
    # time the tolerance verify of the strong net (checker cost proxy)
    t0=time.time(); tolerant(n, b+oets_rounds(n,minR), targets); dt=time.time()-t0
    print(f"n={n:2d} bubble={len(bub):3d} batcher={len(b):3d} B(dupbub)={B:3d} greedy(dupbat)={g:3d} minR={minR} strong={s:3d} | rt={r(B):.3f} rg={r(g):.3f} rs={r(s):.3f} gap_sg={r(s)-r(g):.3f} gap_gt={r(g)-r(B):.3f} | verify={dt:.2f}s")
