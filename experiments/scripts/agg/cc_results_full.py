#!/usr/bin/env python3
"""Fully-labeled dump of the matrix: every model, every benchmark, plain-language roles."""
import json, glob, re
ROOT="/scratch/gpfs/CHIJ/bohan/fs"; FS=f"{ROOT}/FrontierSmith"; TH=f"{ROOT}/ThetaEvolve"

def m5(m,g,k="score"):
    try: return m[g][k]["mean@5"]
    except Exception: return None

fcsale={}
for p in glob.glob(f"{FS}/outputs/cc_eval_*_thinking_32k_both_vllm/summary.json"):
    tag=re.search(r"cc_eval_(.+?)_thinking_32k_both_vllm",p).group(1)
    try: m=json.load(open(p)).get("metrics",{})
    except Exception: continue
    def best(g,k="score"):
        try: return m[g][k]["best@5/mean"]
        except Exception: return None
    fcsale[tag]={"FCSm":m5(m,"frontiercs"),"FCSb":best("frontiercs"),
                 "ALEm":m5(m,"alebench"),"ALEb":best("alebench"),
                 "ALEabs":m5(m,"alebench","overall_absolute_score")}
def theta_load(ttt):
    out={}
    for p in glob.glob(f"{TH}/outputs/cc_eval_theta_*/**/summary.json",recursive=True):
        top=p.split("/outputs/")[1].split("/")[0]
        is_ttt=top.startswith("cc_eval_theta_ttt_")
        if ttt!=is_ttt: continue
        try: d=json.load(open(p))
        except Exception: continue
        tag=d.get("tag");
        if tag and tag.startswith("ttt_"): tag=tag[4:]
        if tag: out[tag]=d.get("best_combined_score")
    return out
theta=theta_load(False); ttt=theta_load(True)

def c(v,br=False,p=2):
    if not isinstance(v,(int,float)): return "   -   "
    return f"{v:6.{p}f}{'!' if br else ' '}"
def row(label,tag):
    r=fcsale.get(tag,{}); a=r.get("ALEabs"); br=isinstance(a,(int,float)) and a<=0
    return (f"  {label:<24}"
            f"{c(r.get('FCSm'))} {c(r.get('FCSb'))} | "
            f"{c(r.get('ALEm'),br,1)} {c(r.get('ALEb'),br,1)} | "
            f"{c(theta.get(tag))} | {c(ttt.get(tag))}")

print("""
NAMING
  q35            = Qwen3.5-9B
  aNN            = STARTING model = NN% instruct + (100-NN)% base
                   (a00 = pure base, a100 = pure instruct, a50 = half/half ...)
  innovonly      = that start model, SFT'd on innovation data ONLY
  innovmaint     = that start model, SFT'd on innovation + maintain data
  soup K x DATA  = weight-average: K*(SFT model) + (1-K)*(START model)   K in {0.5,0.7,0.9}

BENCHMARKS  (all: higher = better).  FCS & ALE each shown as  mean | best :
  mean = mean@5  (average over 5 samples/problem, then over problems)
  best = best@5  (take the BEST of the 5 samples per problem, then average)
  FCS   = FrontierCS    competitive coding
  ALE   = ALE-Bench     performance   ('!' = submissions don't compile, real score = 0)
  Theta = ThetaEvolve   circle-packing  (discovery; single evolutionary run, ~0.96 = failed to beat seed)
  TTT   = TTT-Discover  AC3 autocorrelation  (discovery; single run)
""")
hdr=(f"  {'role':<24}{'FCSmean':>7} {'FCSbest':>7} | {'ALEmean':>7} {'ALEbest':>7} | "
     f"{'Theta':>7} | {'TTT':>6}")
for a,desc in [("a00","100% base, 0% instruct"),("a20","20% instruct, 80% base"),
               ("a50","50% instruct, 50% base"),("a80","80% instruct, 20% base"),
               ("a100","100% instruct (pure)")]:
    print(f"═══════════ START {a}  ({desc}) ═══════════")
    print(hdr)
    print(row("START (no SFT)", f"q35_{a}"))
    print(row("SFT innovonly", f"q35_{a}_innovonly"))
    print(row("SFT innovmaint", f"q35_{a}_innovmaint"))
    for k in ("50","70","90"):
        print(row(f"soup 0.{k} x innovonly", f"q35_{a}_innovonly_soupa{k}"))
    for k in ("50","70","90"):
        print(row(f"soup 0.{k} x innovmaint", f"q35_{a}_innovmaint_soupa{k}"))
    print()
