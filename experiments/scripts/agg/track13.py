import json, glob, os
FS="/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"; TH="/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve"
def fa(tag):
    p=f"{FS}/outputs/cc_eval_{tag}_thinking_32k_both_vllm/summary.json"
    if not os.path.exists(p): return None
    try: m=json.load(open(p))["metrics"]
    except: return None
    return (m["frontiercs"]["score"]["mean@5"], m["frontiercs"]["score"]["best@5/mean"],
            m["alebench"].get("score",{}).get("mean@5",float("nan")), m["alebench"].get("overall_absolute_score",{}).get("mean@5",-1))
def th(tag, ttt=False):
    pre="cc_eval_theta_ttt_" if ttt else "cc_eval_theta_"
    for p in glob.glob(f"{TH}/outputs/{pre}{tag}_*/**/summary.json", recursive=True):
        top=p.split("/outputs/")[1].split("/")[0]
        if ttt and not top.startswith("cc_eval_theta_ttt_"): continue
        if (not ttt) and top.startswith("cc_eval_theta_ttt_"): continue
        try: d=json.load(open(p)); t=d.get("tag","")
        except: continue
        if t.startswith("ttt_"): t=t[4:]
        if t==tag: return d.get("best_combined_score")
    return None
def row(label,tag):
    r=fa(tag)
    if r is None: print(f"  {label:<26} (pending)"); return
    fcs,fcsb,ale,abso=r; brk="!" if abso<=0 else " "
    print(f"  {label:<26} FCS {fcs:6.3f}/{fcsb:6.3f} | ALE {ale:6.1f}{brk} abs={abso/1e6:7.1f}M | Th {th(tag) or 0:.2f} TTT {th(tag,True) or 0:.2f}")
for fam in ["q3","q35"]:
    for a,nm in [("a00","BASE"),("a100","INSTRUCT")]:
        print(f"\n===== {fam} {a} ({nm}) =====")
        row("START", f"{fam}_{a}")
        for data in ["method","methodtraj"]:
            row(f"{data} SFT", f"{fam}_{a}_{data}")
            for p in ["10","20","30","50","70"]:
                row(f"  {data} soup{p}", f"{fam}_{a}_{data}_soupa{p}")
