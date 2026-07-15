#!/usr/bin/env python3
"""Stream rollout dumps -> per arm/step stats + per-problem mean rewards.
Outputs JSON to stdout: {arm: {step: {...stats}}}, plus per-problem file."""
import json, os, sys, collections, statistics

ROOT='/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs/rl_frontiersmith_synth_rollout'
ARMS={'start':'fsx_q35_inst_start','wd03':'fsx_cl_wd03_a10','nom_a5':'fsx_cl_nom_a5','newmt':'fsx_cl_newmt_a10'}

stats={}
perprob={}   # arm -> step -> {gts: mean reward}
for arm,d in ARMS.items():
    stats[arm]={}; perprob[arm]={}
    for step in range(1,21):
        p=f'{ROOT}/{d}/{step}.jsonl'
        if not os.path.exists(p): continue
        rewards=[]; raws=[]; lens=[]; groups=collections.defaultdict(list)
        think_end=0; code_end=0; n=0
        for line in open(p, errors='replace'):
            try: r=json.loads(line)
            except Exception: continue
            n+=1
            rw=r.get('reward_normed', r.get('score',0.0)) or 0.0
            raw=r.get('reward_raw',0.0) or 0.0
            out=r.get('output','')
            rewards.append(rw); raws.append(raw); lens.append(len(out))
            groups[r['gts']].append(rw)
            if '</think>' in out: think_end+=1
            if out.rstrip().endswith('```'): code_end+=1
        if not n: continue
        gz=0; ggrad=0; gsamehi=0
        gmaxes=[]
        pp={}
        for g,rs in groups.items():
            pp[g]=sum(rs)/len(rs)
            gmaxes.append(max(rs))
            if all(x==0 for x in rs): gz+=1
            elif len(set(round(x,6) for x in rs))>1: ggrad+=1
            else: gsamehi+=1
        # non-degenerate = group has >1 distinct reward value => nonzero advantage
        lens_s=sorted(lens)
        rew_s=sorted(rewards)
        stats[arm][step]=dict(
            n=n, ngroups=len(groups),
            reward_mean=sum(rewards)/n,
            reward_median=rew_s[n//2],
            reward_p90=rew_s[int(n*0.9)],
            frac_zero=sum(1 for x in rewards if x==0)/n,
            frac_pos=sum(1 for x in rewards if x>0)/n,
            raw_mean=sum(raws)/n,
            len_chars_mean=sum(lens)/n,
            len_chars_median=lens_s[n//2],
            len_chars_p90=lens_s[int(n*0.9)],
            frac_think_closed=think_end/n,
            frac_ends_code=code_end/n,
            groups_all_zero=gz/len(groups),
            groups_with_gradient=ggrad/len(groups),
            groups_same_nonzero=gsamehi/len(groups),
            group_max_mean=sum(gmaxes)/len(gmaxes),
        )
        perprob[arm][step]=pp

outdir=os.path.dirname(os.path.abspath(__file__))
json.dump(perprob, open(f'{outdir}/rollout_perprob.json','w'))
print(json.dumps(stats))
