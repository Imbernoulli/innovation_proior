#!/usr/bin/env python3
"""H4: per-problem paired diffs, lottery decomposition, bootstrap CIs on FCS."""
import json, random, statistics

M=json.load(open('/tmp/claude-372967/-scratch-gpfs-CHIJ-bohan-fs/3a50c5c5-a73c-4e20-b786-5776707c2962/scratchpad/eval_matrix.json'))
ARMS=['start','wd03','nom_a5','newmt']

def slots(arm,step):
    return M[arm][str(step)]['fcs']['per_prob_slots']

# ---------- 1) base s15 -> s20 gain decomposition ----------
print('=== base (start) s15->s20 per-problem diff, top contributors ===')
s15=slots('start',15); s20=slots('start',20)
pids=sorted(set(s15)|set(s20), key=lambda p:int(p))
diffs=[]
for p in pids:
    a=sum(s15.get(p,[0]*5))/5; b=sum(s20.get(p,[0]*5))/5
    diffs.append((b-a,p,a,b))
diffs.sort(reverse=True)
tot=sum(d[0] for d in diffs)/len(pids)
print(f'total mean diff = {tot:.3f} over {len(pids)} problems')
for d,p,a,b in diffs[:12]:
    v15=[round(x,1) for x in s15.get(p,[0]*5)]
    v20=[round(x,1) for x in s20.get(p,[0]*5)]
    print(f'  prob {p:>4s}: {a:6.2f}->{b:6.2f} (+{d:5.2f})  s15slots={v15} s20slots={v20}')
neg=[x for x in diffs if x[0]<-1]
print('big losers:', [(p, round(d,2)) for d,p,a,b in neg[:8]])

# ---------- 2) lottery classification at s20 for each arm ----------
# lottery problem: per-problem mean driven by <=1 nonzero slot with slot value>=50
print('\n=== s20 composition per arm: lottery vs broad ===')
for arm in ARMS:
    sl=slots(arm,20)
    lottery=0.0; broad=0.0; total=0.0
    nlot=0
    for p,v in sl.items():
        m=sum(v)/5
        total+=m
        nz=[x for x in v if x>0]
        if len(nz)==1 and nz[0]>=50:
            lottery+=m; nlot+=1
        else:
            broad+=m
    n=len(sl)
    print(f'{arm:8s}: overall={total/n:6.3f}  lottery_contrib={lottery/n:6.3f} ({100*lottery/total:4.1f}%, {nlot} probs)  broad={broad/n:6.3f}')

# ---------- 3) bootstrap CI (resample problems) for s20 pairwise diffs ----------
print('\n=== bootstrap (10000 resamples over problems) s20 pairwise ===')
random.seed(0)
def overall_from(sl, pids_sample):
    return sum(sum(sl[p])/5 for p in pids_sample)/len(pids_sample)
B=10000
sl_all={arm:slots(arm,20) for arm in ARMS}
common=sorted(set.intersection(*[set(sl_all[a]) for a in ARMS]), key=lambda p:int(p))
print('common problems:', len(common))
for a in ARMS:
    for b in ARMS:
        if a>=b: continue
        ds=[]
        for _ in range(B):
            samp=[common[random.randrange(len(common))] for _ in range(len(common))]
            ds.append(overall_from(sl_all[a],samp)-overall_from(sl_all[b],samp))
        ds.sort()
        lo,hi=ds[int(B*0.025)],ds[int(B*0.975)]
        mean=sum(ds)/B
        sig='*' if lo>0 or hi<0 else ' '
        print(f'{a:8s} - {b:8s}: mean={mean:+6.3f}  95%CI=[{lo:+6.3f},{hi:+6.3f}] {sig}')

# also bootstrap the s15->s20 gain of base (paired over problems)
print('\n=== bootstrap base gain s15->s20 (paired) ===')
gain=[(sum(s20.get(p,[0]*5))/5 - sum(s15.get(p,[0]*5))/5) for p in common]
ds=[]
for _ in range(B):
    samp=[gain[random.randrange(len(gain))] for _ in range(len(gain))]
    ds.append(sum(samp)/len(samp))
ds.sort()
print(f'mean={sum(gain)/len(gain):+.3f} 95%CI=[{ds[int(B*0.025)]:+.3f},{ds[int(B*0.975)]:+.3f}]')

# ---------- 4) per-arm per-step top-problem trajectories (the 7 named problems) ----------
print('\n=== named problems trajectory (per-problem mean by step) ===')
NAMED=['12','35','147','114','99','162','68']
for p in NAMED:
    row=f'prob {p:>4s}: '
    for arm in ARMS:
        vals=[]
        for s in [0,5,10,15,20]:
            sl=slots(arm,s)
            vals.append(sum(sl.get(p,[0]*5))/5)
        row+=f'{arm}=' + '/'.join(f'{v:.0f}' for v in vals)+'  '
    print(row)
