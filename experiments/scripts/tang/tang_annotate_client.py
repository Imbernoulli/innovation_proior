#!/usr/bin/env python3
"""Scholarly annotation for the Tang & Yang replication (their S1.2 'Output standardization' prompt, verbatim).

Input jsonl rows: {"doc_id", "kind": "paper"|"generated_idea", "text"}.
Output jsonl rows: {"doc_id", "kind", "analysis": {...}, "keywords": [...], "unparsed": bool, ...}.
Resumable: rows whose doc_id is already in --out are skipped. The annotator runs with thinking off
(chat_template_kwargs.enable_thinking=false); if the server ignores that, the parser still takes the
last JSON object after any </think>.
"""
import argparse, json, os, re, sys, time, concurrent.futures as cf
import requests

SYSTEM = ("You are a careful scholarly annotator. Given a research manuscript, write a concise scholarly analysis covering "
          "Aim, Motivation, Questions addressed, Method, Evaluation metrics, Findings, Contributions, Limitations, and Future work. "
          "Finally extract 5-12 concise scholarly keywords grounded in that analysis. Return exactly one valid JSON object, "
          "with no markdown, no prose outside the JSON, and no missing JSON fields.")
USER = """Document kind: {kind}
Document id: {doc_id}
Please proceed to conduct a scholarly analysis of the provided research manuscript. Your analysis should encapsulate the core components of the study as delineated in the enumeration below:
Aim: What is the aim of the study?
Motivation: What is the motivation of the study?
Questions addressed: What question does this study address?
Methods: What methods does the study use to solve the question?
Evaluation metrics: What evaluation metrics are used in this study?
Findings: What does the study find?
Contributions: What are the contributions of this study?
Limitations: What are the limitations of this study?
Future work: What is the future work of this study?
Subsequently, organize the distilled information into a structured JSON format, omitting any supplementary explanations. Return exactly one JSON object with this structure:
{{"analysis": {{"Aim": "...", "Motivation": "...", "Questions addressed": "...", "Method": "...", "Evaluation metrics": "...", "Findings": "...", "Contributions": "...", "Limitations": "...", "Future work": "..."}}, "keywords": ["...", "..."]}}
Rules:
- Output JSON only; do not wrap it in markdown fences. Replace every "..." placeholder with real content from the manuscript; never output "..." itself.
- Include every analysis field exactly as shown above.
- keywords must be a non-empty list of 5-12 short scholarly noun phrases.
- If a field is uncertain, write a brief best-effort value rather than omitting it.
Research manuscript:
{text}"""
FIELDS = ["Aim", "Motivation", "Questions addressed", "Method", "Evaluation metrics", "Findings", "Contributions", "Limitations", "Future work"]
_ALIAS = {"aims": "Aim", "objective": "Aim", "objectives": "Aim", "methods": "Method", "methodology": "Method", "questions": "Questions addressed",
          "question addressed": "Questions addressed", "questions addressed": "Questions addressed", "research questions": "Questions addressed",
          "evaluation metric": "Evaluation metrics", "evaluation": "Evaluation metrics", "metrics": "Evaluation metrics", "finding": "Findings", "results": "Findings",
          "contribution": "Contributions", "limitation": "Limitations", "future works": "Future work", "future directions": "Future work"}
def canon(k):
    kk = re.sub(r"[^a-z ]", " ", str(k).lower()); kk = re.sub(r"\s+", " ", kk).strip()
    for f in FIELDS:
        if kk == f.lower(): return f
    return _ALIAS.get(kk)

def _unesc(s):
    try: return json.loads('"' + s + '"')
    except Exception: return s
def repair(body):
    """Regex fallback for near-JSON (missing commas / stray braces): pull each field's string value and the keywords list."""
    an = {}
    for m in re.finditer(r'"([^"\n]{2,40})"\s*:\s*"((?:[^"\\]|\\.)*)"', body):
        c = canon(m.group(1))
        if c and c not in an: an[c] = _unesc(m.group(2)).strip()
    km = re.search(r'"key\s*words?"\s*:\s*\[(.*?)\]', body, flags=re.S | re.I)
    kw = [_unesc(x) for x in re.findall(r'"((?:[^"\\]|\\.)*)"', km.group(1))] if km else []
    if len(an) < 5 or not kw: return None
    an = {k: an.get(k, "") for k in FIELDS}
    kw = [str(k).strip() for k in kw if str(k).strip() and str(k).strip() not in ("...", "…")][:12]
    if len(kw) < 3 or sum(1 for v in an.values() if v in ("", "...", "…")) > 2: return None
    if not an["Questions addressed"] or not an["Method"]: return None
    return {"analysis": an, "keywords": kw, "repaired": True}

def parse(txt):
    body = txt.split("</think>")[-1]
    body = re.sub(r"^```(?:json)?|```$", "", body.strip(), flags=re.M).strip()
    # take the outermost {...}
    i, j = body.find("{"), body.rfind("}")
    if i < 0 or j <= i: return None
    try: o = json.loads(body[i:j+1])
    except Exception: return repair(body[i:j+1])
    if not isinstance(o, dict): return None
    o = {str(k): v for k, v in o.items()}
    an = next((v for k, v in o.items() if k.strip().lower() in ("analysis", "scholarly analysis", "scholarly_analysis") and isinstance(v, dict)), None)
    if an is None and sum(1 for k in o if canon(k)) >= 5: an = o          # fields at top level, no "analysis" wrapper
    kw = next((v for k, v in o.items() if k.strip().lower() in ("keywords", "keyword", "key words")), None)
    if kw is None and isinstance(an, dict): kw = next((v for k, v in an.items() if k.strip().lower() in ("keywords", "keyword", "key words")), None)
    if isinstance(kw, str): kw = [x.strip() for x in re.split(r"[;,\n]", kw) if x.strip()]
    if not isinstance(an, dict) or not isinstance(kw, list) or not kw: return None
    def s(v):
        if isinstance(v, (list, tuple)): return "; ".join(s(x) for x in v)
        if isinstance(v, dict): return "; ".join(f"{k}: {s(x)}" for k, x in v.items())
        return str(v)
    an2 = {}
    for k, v in an.items():
        c = canon(k)
        if c and c not in an2: an2[c] = s(v).strip()
    an = {k: an2.get(k, "") for k in FIELDS}
    kw = [str(k).strip() for k in kw if str(k).strip() and str(k).strip() not in ("...", "…")][:12]
    if len(kw) < 3 or sum(1 for v in an.values() if v in ("", "...", "…")) > 2: return None   # schema echoed back / placeholders
    if not an["Questions addressed"] or not an["Method"] or an["Method"] in ("...", "…"): return None
    return {"analysis": an, "keywords": kw}

def one(a, row):
    payload = {"model": a.model, "messages": [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": USER.format(kind=row["kind"], doc_id=row["doc_id"], text=row["text"][:12000])}],
               "temperature": 0.0, "max_tokens": a.max_tokens, "n": 1,
               "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    for attempt in range(4):
        try:
            r = requests.post(f"{a.base_url}/chat/completions", json=payload, timeout=a.timeout); r.raise_for_status()
            ch = r.json()["choices"][0]; txt = ch["message"].get("content") or ""
            p = parse(txt)
            rec = {"doc_id": row["doc_id"], "kind": row["kind"], "finish_reason": ch.get("finish_reason"),
                   "completion_tokens": r.json().get("usage", {}).get("completion_tokens"), "seconds": round(time.time() - t0, 2),
                   "unparsed": p is None}
            if p: rec.update(p)
            else: rec["text_tail"] = txt[-500:]; rec["text_full"] = txt
            return rec
        except Exception as e:
            if attempt == 3: return {"doc_id": row["doc_id"], "kind": row["kind"], "unparsed": True, "error": f"{type(e).__name__}: {e}"[:300]}
            time.sleep(3 * (attempt + 1))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=1500); ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=600); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.docs)]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)["doc_id"])
            except Exception: pass
    todo = [r for r in rows if r["doc_id"] not in done][a.shard::a.nshard]
    if a.limit: todo = todo[:a.limit]
    print(f"[annotate] {len(rows)} docs, {len(done)} done, {len(todo)} to do", flush=True)
    n_ok = n_bad = 0; t0 = time.time()
    with open(a.out, "a") as f, cf.ThreadPoolExecutor(a.concurrency) as ex:
        for i, rec in enumerate(ex.map(lambda r: one(a, r), todo), 1):
            f.write(json.dumps(rec) + "\n")
            if rec.get("unparsed"): n_bad += 1
            else: n_ok += 1
            if i % 200 == 0: f.flush(); print(f"[annotate] {i}/{len(todo)} ok={n_ok} bad={n_bad} {time.time()-t0:.0f}s", flush=True)
    print(f"[annotate] DONE ok={n_ok} bad={n_bad}", flush=True)
    sys.exit(0 if n_bad <= 0.05 * max(1, n_ok + n_bad) else 2)

if __name__ == "__main__": main()
