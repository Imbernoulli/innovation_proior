#!/usr/bin/env python3
"""GiantsBench similarity judge (OpenRouter).

The rubric below is transcribed verbatim from Figure 12 of arXiv:2604.09793
(`similarity_judge_prompt.png` in the arXiv HTML build) -- the same prompt the
authors used both as the RL reward and as the evaluation metric.

    OPENROUTER_API_KEY=... python judge_giants.py --gen out_giants.jsonl \
        --out judged.jsonl --judge google/gemini-3.1-pro-preview
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benches import extract_insight  # noqa: E402

JUDGE_PROMPT = """Below is a research insight:
<research_insight>
{gold}
</research_insight>
Below is a statement you need to evaluate:
<statement>
{pred}
</statement>
Task: Rate how similar the statement is to the research insight (1-10).

STRICT RULES:
- Similarity requires matching the SAME core idea.
- 'Inspired by', 'motivated by', or 'reasonable extension' != same idea.
- Shared topic or keywords alone != similarity.

Compare explicitly:
1) Key mechanism/method
2) Causal logic/workflow
3) Primary contribution/novelty

Downgrade if the statement:
- Omits the central mechanism
- Generalizes/abstracts the insight
- Proposes a new framework/direction

Scale:
1-2: Unrelated.
3-4: Shares topic but not the actual insight.
5-6: Partial conceptual overlap; misses at least one core mechanism or misaligns assumptions.
7-8: Strong match with only minor differences in mechanisms or assumptions.
9: Near-identical conceptual + causal + motivational mapping; only minor, non-substantive deviations.
10: Perfect: same ideas, same mechanism and roles, same objective/assumptions.
### Output Format
Format your response as follows:
<think>
Explain your reasoning for the rating you chose.
</think>
<rating>a number between 1 and 10</rating>"""

RATING_RE = re.compile(r"<rating>\s*([0-9]+(?:\.[0-9]+)?)\s*</rating>", re.I)


def parse_rating(text: str):
    if not text:
        return None
    m = RATING_RE.findall(text)
    if not m:
        m = re.findall(r"<rating>\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if not m:
        m = re.findall(r"\brating\D{0,10}([0-9]{1,2}(?:\.[0-9])?)\b", text, re.I)
    if not m:
        return None
    try:
        v = float(m[-1])
    except ValueError:
        return None
    return v if 1.0 <= v <= 10.0 else None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True, help="run_gen.py output for --bench giants")
    p.add_argument("--out", required=True)
    p.add_argument("--judge", default="google/gemini-3.1-pro-preview")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                   help="OpenAI-compatible endpoint; point at a local vLLM to judge offline")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--max-judge-tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--ids-from", default=None, help="only judge ids present in this jsonl")
    args = p.parse_args()

    local = "openrouter.ai" not in args.base_url
    key = (
        os.environ.get(args.api_key_env)
        or os.environ.get(args.api_key_env + "_NEW")
        or ("EMPTY" if local else None)
    )
    if not key:
        sys.exit(f"set ${args.api_key_env}")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key=key, timeout=1800.0, max_retries=0)

    rows = [json.loads(l) for l in open(args.gen) if l.strip()]
    keep = None
    if args.ids_from:
        keep = {json.loads(l)["id"] for l in open(args.ids_from) if l.strip()}
        rows = [r for r in rows if r["id"] in keep]
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"[judge] {args.judge} rows={len(rows)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    fh = open(args.out, "a")
    t0 = time.time()
    cnt = {"n": 0, "err": 0, "empty": 0}

    async def one(r):
        pred = extract_insight(r.get("output") or "")
        gold = (r.get("meta") or {}).get("gold") or ""
        if not pred or not gold:
            async with lock:
                cnt["empty"] += 1
                fh.write(json.dumps({"id": r["id"], "rating": None, "reason": "no_insight",
                                     "meta": r.get("meta")}, ensure_ascii=False) + "\n")
                fh.flush()
            return
        prompt = JUDGE_PROMPT.format(gold=gold.strip(), pred=pred.strip())
        last = None
        for k in range(args.retries + 1):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=args.judge,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=args.max_judge_tokens,
                    )
                txt = resp.choices[0].message.content or ""
                rating = parse_rating(txt)
                if rating is None and k < args.retries:
                    last = ValueError("unparseable rating")
                    await asyncio.sleep(2 * (k + 1))
                    continue
                async with lock:
                    cnt["n"] += 1
                    fh.write(json.dumps({"id": r["id"], "rating": rating, "judge": args.judge,
                                         "raw": txt[-1500:], "meta": r.get("meta")},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                    if cnt["n"] % 25 == 0:
                        print(f"[judge] {cnt['n']}/{len(todo)} err={cnt['err']} "
                              f"empty={cnt['empty']} {(time.time()-t0)/60:.1f}min", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                last = e
                await asyncio.sleep(min(60, 3 * (k + 1)))
        async with lock:
            cnt["err"] += 1
            fh.write(json.dumps({"id": r["id"], "rating": None,
                                 "reason": f"{type(last).__name__}: {last}",
                                 "meta": r.get("meta")}, ensure_ascii=False) + "\n")
            fh.flush()

    await asyncio.gather(*(one(r) for r in todo))
    fh.close()
    print(f"[judge] DONE n={cnt['n']} err={cnt['err']} empty={cnt['empty']} "
          f"wall={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
