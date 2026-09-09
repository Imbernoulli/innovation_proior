#!/usr/bin/env python3
"""Turn gen_client samples.jsonl (one dir per TAG) into annotation documents.
Standardized generated-idea document = Title + Short Hypothesis + Abstract (paper S1.2: 'title (or name), hypothesis, and
abstract-like proposal text'). Invalid = no parsable JSON with a non-empty Abstract (paper's validity filter)."""
import json, os, re, sys, glob
S = os.path.dirname(os.path.abspath(__file__)); D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
def parse(txt):
    body = txt.split("</think>")[-1]; body = re.sub(r"^```(?:json)?|```$", "", body.strip(), flags=re.M)
    i, j = body.find("{"), body.rfind("}")
    if i < 0 or j <= i: return None
    try: o = json.loads(body[i:j+1])
    except Exception:
        try: o = json.loads(re.sub(r",\s*}", "}", body[i:j+1]))
        except Exception: return None
    if not isinstance(o, dict) or not str(o.get("Abstract", "")).strip(): return None
    return o
out = open(f"{S}/docs_ideas.jsonl", "w"); stats = {}
for d in sorted(glob.glob(f"{D}/outputs/cc_tang_*/samples.jsonl")):
    tag = d.split("/cc_tang_")[1].split("/")[0]; ok = bad = 0
    for l in open(d):
        r = json.loads(l)
        if r.get("task") != "tang_zero": continue
        o = parse(r.get("text", "") or "")
        if o is None: bad += 1; continue
        ok += 1
        text = f"Title: {o.get('Title') or o.get('Name','')}\n\nHypothesis: {o.get('Short Hypothesis','')}\n\nAbstract: {o.get('Abstract','')}"
        out.write(json.dumps({"doc_id": f"{tag}|{r['id'].split('/')[-1]}|{r['sample_idx']}", "kind": "generated_idea", "text": text,
                              "tag": tag, "seed_id": r["id"].split("/")[-1], "sample_idx": r["sample_idx"], "finish_reason": r.get("finish_reason")}) + "\n")
    stats[tag] = (ok, bad)
for t, (ok, bad) in stats.items(): print(f"{t:28s} valid={ok} invalid={bad} validity={ok/max(1,ok+bad):.3f}")
