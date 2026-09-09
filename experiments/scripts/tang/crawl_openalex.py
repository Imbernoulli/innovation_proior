#!/usr/bin/env python3
"""Tang & Yang (2605.27905) small-scale replication -- data side, via OpenAlex.

Deviations from the paper, all deliberate and recorded in the scorecard:
  * research areas = hand-picked OpenAlex topics (48 = 12 fields x 4), not bibliographic-coupling clusters;
  * corpus = OpenAlex (S2 refuses unauthenticated traffic with 429);
  * seed years t in {2022, 2023} only (follow-on year t+1 needs citation counts to have accumulated);
  * anchors must have >= MIN_CITES citations and >= 15 references so that the 4 related papers can be
    drawn from the anchor's own reference list (same topic, year <= t) -- 'selected using citation'.
Everything else follows S1.2 / S1.3: 5-paper seed set, follow-on = year t+1 papers citing >= 1 seed,
impact pool = same-area human papers published <= t, frontier corpus = field-level papers in t+1.
"""
import json, os, random, sys, time, urllib.request, urllib.parse

S = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(S, "data")
MAILTO = "lyubh22@gmail.com"; BASE = "https://api.openalex.org/works"
YEARS = [2022, 2023]; ANCHORS_PER = 3; MIN_CITES = {2022: 15, 2023: 10}
N_FOLLOW = 30; N_IMPACT = 400; N_FRONTIER = 400
FIELD_ID = {"Medicine":27,"Biology":13,"Engineering":22,"Chemistry":16,"ComputerScience":17,"EnvironmentalScience":23,
            "MaterialsScience":25,"Physics":31,"Mathematics":26,"Economics":20,"Business":14,"SocialSciences":33}
SEL = "id,title,publication_year,cited_by_count,referenced_works,primary_topic,abstract_inverted_index,type,language"
rng = random.Random(20260909)
papers = {}   # id -> record (dedup store)

def get(params, retries=6):
    q = urllib.parse.urlencode({**params, "mailto": MAILTO}, safe="|:,>")
    err = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(BASE + "?" + q, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(2 * (k + 1)); err = e
    print("GIVE UP", q[:200], err, file=sys.stderr); return {"results": [], "meta": {"count": 0}}

def abstract(inv):
    if not inv: return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)

def keep(w, need_abs=True):
    a = abstract(w.get("abstract_inverted_index"))
    if need_abs and len(a.split()) < 40: return None
    pt = (w.get("primary_topic") or {})
    rec = {"id": w["id"].rsplit("/", 1)[-1], "title": w.get("title") or "", "year": w.get("publication_year"),
           "cites": w.get("cited_by_count", 0), "topic": (pt.get("id") or "").rsplit("/", 1)[-1],
           "field": ((pt.get("field") or {}).get("id") or "").rsplit("/", 1)[-1], "abstract": a,
           "refs": [r.rsplit("/", 1)[-1] for r in w.get("referenced_works", [])]}
    papers[rec["id"]] = rec; return rec

def sample(filt, n, seed, need_abs=True):
    out = []
    for page in range(0, n, 200):
        d = get({"filter": filt, "sample": n, "seed": seed, "per-page": min(200, n - page), "page": page // 200 + 1, "select": SEL})
        out += [r for r in (keep(w, need_abs) for w in d["results"]) if r]
        if len(d["results"]) < min(200, n - page): break
    return out

def by_ids(ids):
    out = []
    for i in range(0, len(ids), 50):
        d = get({"filter": "openalex_id:" + "|".join(ids[i:i+50]), "per-page": 50, "select": SEL})
        out += [r for r in (keep(w) for w in d["results"]) if r]
    return out

areas = json.load(open(os.path.join(S, "areas.json")))
seedsets, follow, impact, frontier = [], [], [], []
common = "type:article,language:en,has_abstract:true"
for field, lst in areas.items():
    for T, name in lst:
        pool = sample(f"primary_topic.id:{T},publication_year:2018-2023,cited_by_count:>2,{common}", N_IMPACT, 11)
        impact += [{"area": T, "field": field, "id": p["id"], "year": p["year"], "cites": p["cites"]} for p in pool]
        for t in YEARS:
            cands = sample(f"primary_topic.id:{T},publication_year:{t},cited_by_count:>{MIN_CITES[t]-1},referenced_works_count:>14,{common}", 12, 100 + t)
            rng.shuffle(cands); made = 0
            for a in cands:
                if made >= ANCHORS_PER: break
                refs = by_ids(a["refs"][:120])
                rel = sorted([r for r in refs if r["topic"] == T and r["year"] and r["year"] <= t and r["id"] != a["id"]], key=lambda r: -r["cites"])
                fb = 0
                if len(rel) < 4:
                    extra = sorted([r for r in refs if r not in rel and r["field"] == a["field"] and r["year"] and r["year"] <= t], key=lambda r: -r["cites"])
                    if extra: fb = 1
                    rel += extra[:4 - len(rel)]
                if len(rel) < 4:
                    extra = [r for r in refs if r not in rel and r["year"] and r["year"] <= t]; rel += extra[:4 - len(rel)]; fb = 2
                if len(rel) < 4: continue
                seeds = [a["id"]] + [r["id"] for r in rel[:4]]
                d = get({"filter": f"cites:{'|'.join(seeds)},publication_year:{t+1},{common}", "sample": N_FOLLOW, "seed": 5, "per-page": N_FOLLOW, "select": SEL})
                fo = [r for r in (keep(w) for w in d["results"]) if r]
                sid = f"{T}_{t}_{made}"
                seedsets.append({"seed_id": sid, "area": T, "area_name": name, "field": field, "t": t, "anchor": a["id"], "seeds": seeds, "fallback": fb, "n_follow_total": d["meta"]["count"]})
                follow += [{"seed_id": sid, "area": T, "field": field, "t": t, "id": r["id"], "year": r["year"], "cites": r["cites"]} for r in fo]
                made += 1
            print(f"{field:20s} {T} {name[:40]:40s} t={t} seedsets={made} pool={len(pool)}", flush=True)
    for t in YEARS:
        fc = sample(f"primary_topic.field.id:fields/{FIELD_ID[field]},publication_year:{t+1},cited_by_count:>2,{common}", N_FRONTIER, 300 + t)
        frontier += [{"field": field, "t": t, "id": p["id"]} for p in fc]
        print(f"{field:20s} frontier t+1={t+1}: {len(fc)}", flush=True)

for nm, obj in [("seedsets", seedsets), ("followons", follow), ("impact_pool", impact), ("frontier_corpus", frontier)]:
    with open(os.path.join(OUT, nm + ".jsonl"), "w") as f:
        for r in obj: f.write(json.dumps(r) + "\n")
with open(os.path.join(OUT, "papers.jsonl"), "w") as f:
    for r in papers.values(): r = dict(r); r.pop("refs", None); f.write(json.dumps(r) + "\n")
print("DONE seedsets", len(seedsets), "followons", len(follow), "impact", len(impact), "frontier", len(frontier), "papers", len(papers))
