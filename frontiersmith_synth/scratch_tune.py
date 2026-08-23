import json, random, math
from collections import Counter, deque

N = 1023

def build_balanced(n):
    left=[-1]*n; right=[-1]*n; parent=[-1]*n
    def rec(lo,hi,par):
        if lo>hi: return -1
        mid=(lo+hi)//2
        parent[mid]=par
        left[mid]=rec(lo,mid-1,mid)
        right[mid]=rec(mid+1,hi,mid)
        return mid
    root=rec(0,n-1,-1)
    return root,left,right,parent

class Tree:
    def __init__(self, root,left,right,parent):
        self.root=root; self.left=left[:]; self.right=right[:]; self.parent=parent[:]
    def depth(self,k):
        cur=self.root; d=0
        while cur!=k:
            if k<cur: cur=self.left[cur]
            else: cur=self.right[cur]
            if cur==-1: return None
            d+=1
        return d
    def rotate_up(self,x):
        p=self.parent[x]
        if p==-1: return False
        g=self.parent[p]
        L=self.left; R=self.right; P=self.parent
        if x==L[p]:
            L[p]=R[x]
            if R[x]!=-1: P[R[x]]=p
            R[x]=p; P[p]=x
        else:
            R[p]=L[x]
            if L[x]!=-1: P[L[x]]=p
            L[x]=p; P[p]=x
        P[x]=g
        if g==-1: self.root=x
        elif L[g]==p: L[g]=x
        else: R[g]=x
        return True

# ---------------- generator ----------------
def gen_trace(seed, plan):
    rng=random.Random(seed)
    trace=[]
    for (typ,p) in plan:
        if typ=='ws':
            k=p['k']; length=p['len']
            hot=rng.sample(range(N),k)
            for _ in range(length):
                trace.append(hot[rng.randrange(k)])
        elif typ=='wsdrift':
            k=p['k']; length=p['len']
            hot=rng.sample(range(N),k)
            for t in range(length):
                if t>0 and t%p.get('every',60)==0:
                    hot[rng.randrange(k)]=rng.randrange(N)
                trace.append(hot[rng.randrange(k)])
        elif typ=='seq':
            start=p['start']; length=p['len']; step=p.get('step',1)
            x=start
            for _ in range(length):
                trace.append(x%N); x+=step
        elif typ=='rev':
            start=p['start']; length=p['len']
            x=start; d=1
            for _ in range(length):
                trace.append(x%N); x+=d
                if x>=start+p.get('span',300): d=-1
                if x<=start: d=1
        elif typ=='rand':
            for _ in range(p['len']): trace.append(rng.randrange(N))
    return trace

def instance_plans():
    P=[]
    # 0 WS heavy
    P.append([('ws',{'k':6,'len':700}),('rand',{'len':150}),('ws',{'k':5,'len':600})])
    # 1 WS heavy small set
    P.append([('ws',{'k':4,'len':800}),('ws',{'k':6,'len':700})])
    # 2 seq heavy (TRAP)
    P.append([('seq',{'start':0,'len':1023}),('seq',{'start':0,'len':1023})])
    # 3 seq heavy (TRAP)
    P.append([('rand',{'len':200}),('seq',{'start':100,'len':900,'step':1}),('seq',{'start':100,'len':900})])
    # 4 mixed
    P.append([('ws',{'k':5,'len':500}),('seq',{'start':0,'len':600}),('ws',{'k':5,'len':500})])
    # 5 reversal heavy (TRAP-ish)
    P.append([('rev',{'start':0,'len':1000,'span':400}),('rev',{'start':300,'len':800,'span':400})])
    # 6 WS drift
    P.append([('wsdrift',{'k':6,'len':900,'every':50}),('ws',{'k':5,'len':500})])
    # 7 mixed dispersed
    P.append([('rand',{'len':700}),('ws',{'k':6,'len':700})])
    # 8 seq step (TRAP)
    P.append([('seq',{'start':0,'len':900,'step':7}),('seq',{'start':3,'len':900,'step':7})])
    # 9 mixed balanced
    P.append([('ws',{'k':5,'len':600}),('rand',{'len':300}),('seq',{'start':50,'len':500}),('ws',{'k':4,'len':400})])
    return P

def make_instances():
    out=[]
    root,left,right,parent=build_balanced(N)
    for i,pl in enumerate(instance_plans()):
        tr=gen_trace(7000+i, pl)
        out.append({'root':root,'left':left,'right':right,'trace':tr})
    return out

# ---------------- scoring ----------------
def cost_of_plan(inst, rotations):
    t=Tree(inst['root'],inst['left'],inst['right'],inst['parent'] if 'parent' in inst else _parent(inst))
    trace=inst['trace']; T=len(trace)
    tot=0
    for i in range(T):
        rl=rotations[i] if i<len(rotations) else []
        for x in rl:
            if not t.rotate_up(x): return None
            tot+=1
        d=t.depth(trace[i])
        if d is None: return None
        tot+= d+1
    return tot

def _parent(inst):
    # reconstruct parent from left/right/root
    n=len(inst['left']); parent=[-1]*n
    for i in range(n):
        if inst['left'][i]!=-1: parent[inst['left'][i]]=i
        if inst['right'][i]!=-1: parent[inst['right'][i]]=i
    return parent

def baseline(inst):
    return cost_of_plan(inst, [])

# ---------------- policies ----------------
def policy_trivial(inst):
    return []

def policy_greedy(inst):  # move-to-root of previous accessed key
    t=Tree(inst['root'],inst['left'],inst['right'],_parent(inst))
    trace=inst['trace']; T=len(trace)
    rots=[]
    prev=None
    for i in range(T):
        rl=[]
        if prev is not None:
            d=t.depth(prev)
            for _ in range(d):
                t.rotate_up(prev); rl.append(prev)
        rots.append(rl)
        # account the search on the (rotated) tree to advance prev
        prev=trace[i]
    return rots

def policy_strong(inst, W=64, period=32, conc=14, topk=6, target=2, fmin=3, horizon=180):
    t=Tree(inst['root'],inst['left'],inst['right'],_parent(inst))
    trace=inst['trace']; T=len(trace)
    rots=[[] for _ in range(T)]
    win=deque(); cnt=Counter()
    for i in range(T):
        # phase-detect / restructure decision BEFORE search i (uses only past window)
        if i>0 and i%period==0 and len(win)>=W//2:
            distinct=len(cnt)
            if distinct<=conc:
                # working-set phase: bring the hottest keys up, metered by benefit
                hot=[k for k,c in cnt.most_common(topk) if c>=fmin]
                # order by current depth descending so deepest gains first
                hot.sort(key=lambda k:-(t.depth(k) or 0))
                for k in hot:
                    d=t.depth(k)
                    if d is None: continue
                    rate=cnt[k]/max(1,len(win))
                    # amortized benefit: projected accesses * depth-reduction vs rotation cost
                    proj=rate*horizon
                    gain=proj*(d-target)
                    cost=(d-target)
                    if d>target and gain>cost:
                        rl=rots[i]
                        while (t.depth(k) or 0)>target:
                            kk=k
                            t.rotate_up(kk); rl.append(kk)
        # search i
        t.depth(trace[i])
        win.append(trace[i]); cnt[trace[i]]+=1
        if len(win)>W:
            old=win.popleft(); cnt[old]-=1
            if cnt[old]==0: del cnt[old]
    return rots

def report():
    insts=make_instances()
    tiers={'trivial':policy_trivial,'greedy':policy_greedy,'strong':policy_strong}
    vecs={}
    for name,pol in tiers.items():
        vec=[]
        for inst in insts:
            b=baseline(inst)
            r=pol(inst)
            obj=cost_of_plan(inst,r)
            if obj is None:
                vec.append(0.0); continue
            rr=min(1.0, 0.1*b/max(obj,1e-12))
            vec.append(rr)
        vecs[name]=vec
        print(f"{name:8s} mean={sum(vec)/len(vec):.4f}  vec={[round(x,3) for x in vec]}")
    print("greedy-trivial=",sum(vecs['greedy'])/10-sum(vecs['trivial'])/10)
    print("strong-greedy =",sum(vecs['strong'])/10-sum(vecs['greedy'])/10)

if __name__=='__main__':
    report()
