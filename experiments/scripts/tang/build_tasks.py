#!/usr/bin/env python3
"""From data/seedsets.jsonl + data/papers.jsonl build
  (1) gentasks_tang.jsonl -- zero-shot idea-generation tasks (paper's Zero-shot prompt, verbatim; system folded into user)
  (2) docs_papers.jsonl   -- every human paper (seeds, follow-ons, impact pool, frontier corpus) as an annotation document."""
import json, os
def tr(a, n=350):
    """OpenAlex 'abstracts' occasionally carry full text; cap at n words so prompts stay ~<=2.5K tokens."""
    return " ".join(str(a).split()[:n])
S = os.path.dirname(os.path.abspath(__file__)); DD = os.path.join(S, "data")
papers = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DD}/papers.jsonl")}
seedsets = [json.loads(l) for l in open(f"{DD}/seedsets.jsonl")]
SYS = ("You are an experienced researcher. Given a research topic and a set of relevant papers, propose exactly one novel, feasible research idea. "
       "Output requirements: emit ONLY a single JSON object that matches the schema given by the user. Begin your answer with '{' and end with '}'.")
SCHEMA = ('{"Name": "<short snake_case identifier>", "Title": "<full paper title>", "Short Hypothesis": "<one sentence core claim>", '
          '"Related Work": "<how this differs from the provided papers>", "Abstract": "<~150 word abstract>", '
          '"Experiments": "<key experiments needed to validate the idea>", "Risk Factors and Limitations": "<main risks or limitations>"}')
n = 0
with open(f"{S}/gentasks_tang.jsonl", "w") as f:
    for s in seedsets:
        ctx = "\n\n".join(f"{i+1}. Title: {papers[p]['title']}\nAbstract: {tr(papers[p]['abstract'])}" for i, p in enumerate(s["seeds"]) if p in papers)
        prompt = (f"{SYS}\n\n{ctx}\n\n— Propose ONE novel research idea grounded in the literature above. "
                  f"Reply with a single JSON object matching this schema:\n{SCHEMA}")
        f.write(json.dumps({"task": "tang_zero", "id": f"tang_zero/{s['seed_id']}", "kind": "gen", "keyword": s["area_name"], "prompt": prompt}) + "\n"); n += 1
print("gen tasks", n)
ids = set()
for s in seedsets: ids.update(s["seeds"])
for nm in ("followons", "impact_pool", "frontier_corpus"):
    for l in open(f"{DD}/{nm}.jsonl"): ids.add(json.loads(l)["id"])
with open(f"{S}/docs_papers.jsonl", "w") as f:
    m = 0
    for pid in sorted(ids):
        p = papers.get(pid)
        if not p: continue
        f.write(json.dumps({"doc_id": pid, "kind": "paper", "text": f"Title: {p['title']}\n\nAbstract: {tr(p['abstract'])}"}) + "\n"); m += 1
print("paper docs", m, "of", len(ids), "ids")
