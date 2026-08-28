#!/usr/bin/env python3
"""Lit2Test pairwise judge -- the benchmark's OWN protocol.

Lit2Test has no gold answer: two anonymous six-field proposals for the same
literature context are compared head to head.  The prompt below is a VERBATIM
copy of `build_prompt` in the benchmark's `run_lit2test_v02_pairwise.py`.

Both orders are judged and only ORDER-STABLE verdicts enter the win rate (the
benchmark's own fold); order-sensitive pairs and ties are reported separately.
The repo also ships frozen proposals from four frontier models on the same 200
contexts, so `--opponent-file` can point at those to get a head-to-head number
against DeepSeek-V3.2 / GLM-5 / GPT-5.2 / Sonnet-4.6 instead of only vs base.

  python judge_lit2test.py --a gen_ours.jsonl --b gen_base.jsonl \
      --contexts .cache/taste_eval/lit2test-benchmark --out judged.jsonl \
      --judge judge9b --base-url http://127.0.0.1:PORT/v1
"""
from __future__ import annotations
import argparse, asyncio, glob, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benches import LIT2TEST_FIELDS, extract_proposal  # noqa: E402


def build_prompt(context, answer_a, answer_b) -> str:
    return f"""You are judging two anonymous Lit2Test answers for the same literature context.

The original task: derive one minimal falsifiable next-step test from the provided paper neighborhood. Prefer the answer that is more grounded in the provided papers, uses cross-paper tension, gives a specific hypothesis, proposes a minimal feasible test, matches metrics to mechanisms, and states clear supporting/falsifying outcomes.

Important judging rules:
- The two answers are anonymous. Do not infer or discuss model identity.
- Judge relative quality only for this specific context.
- Penalize generic research proposals that would fit many unrelated paper groups.
- Penalize large unfocused experimental programs; reward a minimal decisive test.
- If both are truly equivalent, choose "tie".

Return valid JSON only with this schema:
{{
  "pair_id": "...",
  "winner": "A" | "B" | "tie",
  "confidence": "low" | "medium" | "high",
  "main_reason": "...",
  "weakness_a": "...",
  "weakness_b": "..."
}}

Context:
```json
{json.dumps(context, ensure_ascii=False, indent=2)}
```

Answer A:
```json
{json.dumps(answer_a, ensure_ascii=False, indent=2)}
```

Answer B:
```json
{json.dumps(answer_b, ensure_ascii=False, indent=2)}
```
"""


def load_gen(path):
    out = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        p = extract_proposal(r.get("output") or "")
        if p:
            out[r["id"]] = p
    return out


def load_frozen(path):
    out = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        out[r["context_id"]] = {k: str(r.get(k, "")) for k in LIT2TEST_FIELDS}
    return out


def load_contexts(root):
    ctx = {}
    for f in sorted(glob.glob(os.path.join(root, "data", "lit2test_v02_*_contexts.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                ctx[r["context_id"]] = {
                    k: r.get(k) for k in
                    ("context_id", "field", "research_context", "open_problem",
                     "resource_constraint", "task_instruction", "papers")
                }
    return ctx


def parse_verdict(text):
    t = re.sub(r"```(?:json)?", "", text or "")
    i = t.find("{")
    while i >= 0:
        d = 0
        for j in range(i, len(t)):
            if t[j] == "{":
                d += 1
            elif t[j] == "}":
                d -= 1
                if d == 0:
                    try:
                        o = json.loads(t[i:j + 1])
                    except json.JSONDecodeError:
                        break
                    w = str(o.get("winner", "")).strip().lower()
                    if w in ("a", "b", "tie"):
                        return w
                    break
        i = t.find("{", i + 1)
    m = re.findall(r'"winner"\s*:\s*"(a|b|tie)"', t, re.I)
    return m[-1].lower() if m else None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--b-frozen", action="store_true", help="--b is a frozen results/proposals file")
    p.add_argument("--contexts", required=True); p.add_argument("--out", required=True)
    p.add_argument("--judge", required=True); p.add_argument("--base-url", required=True)
    p.add_argument("--concurrency", type=int, default=12); p.add_argument("--retries", type=int, default=3)
    a = p.parse_args()
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", timeout=1800.0, max_retries=0)
    A = load_gen(a.a)
    B = load_frozen(a.b) if a.b_frozen else load_gen(a.b)
    C = load_contexts(a.contexts)
    ids = sorted(set(A) & set(B) & set(C))
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [(i, o) for i in ids for o in ("AB", "BA") if f"{i}#{o}" not in done]
    print(f"[lit2test] contexts={len(ids)} judgements todo={len(todo)}", flush=True)
    sem = asyncio.Semaphore(a.concurrency); lock = asyncio.Lock()
    fh = open(a.out, "a"); t0 = time.time(); n = [0]

    async def one(cid, order):
        x, y = (A[cid], B[cid]) if order == "AB" else (B[cid], A[cid])
        prompt = build_prompt(C[cid], x, y)
        for k in range(a.retries + 1):
            try:
                async with sem:
                    r = await client.chat.completions.create(
                        model=a.judge, messages=[{"role": "user", "content": prompt}],
                        temperature=0.0, max_tokens=2048)
                v = parse_verdict(r.choices[0].message.content or "")
                if v is None and k < a.retries:
                    await asyncio.sleep(2); continue
                async with lock:
                    n[0] += 1
                    fh.write(json.dumps({"id": f"{cid}#{order}", "context_id": cid,
                                         "order": order, "winner": v}) + "\n"); fh.flush()
                    if n[0] % 50 == 0:
                        print(f"[lit2test] {n[0]}/{len(todo)} {(time.time()-t0)/60:.1f}min", flush=True)
                return
            except Exception:
                await asyncio.sleep(3 * (k + 1))
        async with lock:
            fh.write(json.dumps({"id": f"{cid}#{order}", "context_id": cid,
                                 "order": order, "winner": None}) + "\n"); fh.flush()

    await asyncio.gather(*(one(c, o) for c, o in todo))
    fh.close()
    # fold: A wins in AB means winner==A; in BA means winner==B
    rec = {}
    for l in open(a.out):
        r = json.loads(l); rec.setdefault(r["context_id"], {})[r["order"]] = r["winner"]
    stable = awin = bwin = tie = sens = 0
    for cid, d in rec.items():
        if len(d) < 2 or None in d.values():
            continue
        a_ab = d["AB"] == "a"; a_ba = d["BA"] == "b"
        t_ab = d["AB"] == "tie"; t_ba = d["BA"] == "tie"
        if t_ab and t_ba:
            tie += 1; stable += 1
        elif a_ab == a_ba and not (t_ab or t_ba):
            stable += 1
            if a_ab: awin += 1
            else: bwin += 1
        else:
            sens += 1
    dec = awin + bwin
    print(f"\n[lit2test] order-stable {stable}/{len(rec)}  order-sensitive {sens}  tie {tie}")
    if dec:
        print(f"[lit2test] A win-rate over decided stable pairs: {awin}/{dec} = {awin/dec:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
