#!/usr/bin/env python3
"""Eval-side analysis: dedup samples (last wins), strict scoring (missing/error=0),
per-problem matrices, truncation curves. Writes eval_matrix.json + prints summary JSON."""
import json, os, glob, collections

OUT='/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs'
ARMS=['start','wd03','nom_a5','newmt']
STEP0={'start':'cc_eval_clean_start_thinking_32k_both_vllm',
       'wd03':'cc_eval_clnom_wd03_a10_thinking_32k_both_vllm',
       'nom_a5':'cc_eval_clean_clean_nomaintain_wd01_a5_thinking_32k_both_vllm',
       'newmt':'cc_eval_clnom_newmt_a10_thinking_32k_both_vllm'}
RLDIR={'start':'cc_eval_rlfsx_q35_inst_start_step{s}_thinking_32k_both_vllm',
       'wd03':'cc_eval_rlfsx_cl_wd03_a10_step{s}_thinking_32k_both_vllm',
       'nom_a5':'cc_eval_rlfsx_cl_nom_a5_step{s}_thinking_32k_both_vllm',
       'newmt':'cc_eval_rlfsx_cl_newmt_a10_step{s}_thinking_32k_both_vllm'}
STEP0_RES={'start':'cc_eval_q35_inst_start_research_research_thinking_32k_vllm',
       'wd03':'cc_eval_clnom_wd03_a10_research_thinking_32k_vllm',
       'nom_a5':'cc_eval_clean_clean_nomaintain_wd01_a5_research_thinking_32k_vllm',
       'newmt':'cc_eval_clnom_newmt_a10_research_thinking_32k_vllm'}
RLDIR_RES={'start':'cc_eval_rlfsx_q35_inst_start_step{s}_research_thinking_32k_vllm',
       'wd03':'cc_eval_rlfsx_cl_wd03_a10_step{s}_research_thinking_32k_vllm',
       'nom_a5':'cc_eval_rlfsx_cl_nom_a5_step{s}_research_thinking_32k_vllm',
       'newmt':'cc_eval_rlfsx_cl_newmt_a10_step{s}_research_thinking_32k_vllm'}

def load_dir(d):
    """dedup last-wins keyed (data_source, problem_idx, sample_idx)"""
    samples={}
    for f in sorted(glob.glob(f'{d}/shard_*/samples.jsonl')):
        for line in open(f, errors='replace'):
            try: r=json.loads(line)
            except Exception: continue
            k=(r['data_source'], str(r['problem_idx']), str(r['sample_idx']))
            samples[k]=r
    return samples

def analyze(samples, source_filter):
    per_prob=collections.defaultdict(dict)  # pid -> sidx -> record
    for (ds,pid,sidx),r in samples.items():
        if ds!=source_filter: continue
        per_prob[pid][sidx]=r
    prob_scores={}; nerr=0; ntrunc=0; ntot=0; cts=[]
    per_prob_slots={}
    for pid,slots in per_prob.items():
        vals=[]
        for sidx in ['0','1','2','3','4']:
            r=slots.get(sidx)
            if r is None or r.get('error'):
                vals.append(0.0)
                if r is not None and r.get('error'): nerr+=1
                ntot+=1
                continue
            m=r.get('metrics') or {}
            rw=m.get('reward',0.0) or 0.0
            vals.append(rw)
            ct=r.get('completion_tokens')
            if ct is not None:
                cts.append(ct)
                if ct>=32768: ntrunc+=1
            ntot+=1
        prob_scores[pid]=sum(vals)/5.0
        per_prob_slots[pid]=vals
    if not prob_scores: return None
    cts.sort()
    overall=sum(prob_scores.values())/len(prob_scores)
    return dict(overall=overall, nprob=len(prob_scores), nerr=nerr, ntot=ntot,
                trunc_rate=ntrunc/max(1,ntot),
                ct_median=cts[len(cts)//2] if cts else None,
                ct_mean=sum(cts)/len(cts) if cts else None,
                per_prob=prob_scores, per_prob_slots=per_prob_slots)

results={}
matrix={}   # arm -> step -> {fcs:..., ale:..., research:...}
for arm in ARMS:
    matrix[arm]={}
    for s in [0,5,10,15,20]:
        entry={}
        d=os.path.join(OUT, STEP0[arm] if s==0 else RLDIR[arm].format(s=s))
        if os.path.isdir(d):
            samples=load_dir(d)
            fcs=analyze(samples,'frontiercs')
            ale=analyze(samples,'alebench')
            entry['fcs']=fcs; entry['ale']=ale
        dr=os.path.join(OUT, STEP0_RES[arm] if s==0 else RLDIR_RES[arm].format(s=s))
        if os.path.isdir(dr):
            rs=load_dir(dr)
            srcs={k[0] for k in rs}
            res=analyze(rs,'frontiercs_research')
            entry['research']=res
        matrix[arm][s]=entry

outdir=os.path.dirname(os.path.abspath(__file__))
json.dump(matrix, open(f'{outdir}/eval_matrix.json','w'))

# compact summary
summ={}
for arm in ARMS:
    summ[arm]={}
    for s,e in matrix[arm].items():
        row={}
        for bench in ['fcs','research']:
            b=e.get(bench)
            if b:
                row[bench]=dict(overall=round(b['overall'],3), nerr=b['nerr'],
                                trunc=round(b['trunc_rate'],3), ct_med=b['ct_median'])
        summ[arm][s]=row
print(json.dumps(summ,indent=1))
