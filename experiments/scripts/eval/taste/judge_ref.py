#!/usr/bin/env python3
"""Reference-based LM judge for the generation-side benchmarks.

`--task giants` reproduces arXiv:2604.09793 Figure 12 verbatim.  `--task abgen`
and `--task hypoarena` reuse the same comparison structure with the object of
comparison changed (an ablation design / a hypothesis instead of an insight):
those two rubrics are OUR ADAPTATION, not the benchmarks' official metrics
(AbGen's is a multi-aspect reference judge, HypoArena's is HypoEval), so their
absolute numbers are only meaningful against the arms in this same run.

    python judge_ref.py --task abgen --gen gen.jsonl --out judged.jsonl \
        --judge judge9b --base-url http://127.0.0.1:PORT/v1
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
from benches import extract_insight, extract_tagged  # noqa: E402

HEAD = {
    "giants": ("research insight", "research_insight", "statement"),
    "abgen": ("ablation study design", "reference_design", "design"),
    "hypoarena": ("research hypothesis", "reference_hypothesis", "hypothesis"),
    "prescience": ("paper contribution (title + abstract)", "reference_paper", "prediction"),
}

RUBRIC_TAIL = """Task: Rate how similar the statement is to the {obj} (1-10).

STRICT RULES:
- Similarity requires matching the SAME core content.
- 'Inspired by', 'motivated by', or 'reasonable extension' != same content.
- Shared topic or keywords alone != similarity.

Compare explicitly:
{axes}

Downgrade if the statement:
- Omits the central element
- Generalizes/abstracts it away
- Substitutes a different framework/direction

Scale:
1-2: Unrelated.
3-4: Shares topic but not the actual content.
5-6: Partial overlap; misses at least one core element or misaligns assumptions.
7-8: Strong match with only minor differences.
9: Near-identical; only minor, non-substantive deviations.
10: Perfect: same content, same roles, same objective/assumptions.
### Output Format
Format your response as follows:
<think>
Explain your reasoning for the rating you chose.
</think>
<rating>a number between 1 and 10</rating>"""

AXES = {
    "giants": "1) Key mechanism/method\n2) Causal logic/workflow\n3) Primary contribution/novelty",
    "abgen": "1) What is varied and what is held fixed\n2) Which comparisons/baselines are run and on what data\n3) Which metrics decide the outcome",
    "hypoarena": "1) The claim being asserted and its mechanism\n2) The entities and conditions it is about\n3) What evidence would settle it",
    "prescience": "1) The problem taken on and why it is open\n2) The method or construction introduced\n3) What is established or measured",
}

RATING_RE = re.compile(r"<rating>\s*([0-9]+(?:\.[0-9]+)?)\s*</rating>", re.I)


def parse_rating(text: str):
    if not text:
        return None
    m = RATING_RE.findall(text) or re.findall(r"<rating>\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if not m:
        m = re.findall(r"\brating\D{0,10}([0-9]{1,2}(?:\.[0-9])?)\b", text, re.I)
    if not m:
        return None
    try:
        v = float(m[-1])
    except ValueError:
        return None
    return v if 1.0 <= v <= 10.0 else None


def build_prompt(task: str, gold: str, pred: str) -> str:
    if task == "giants":
        # MUST be the paper's verbatim Figure-12 rubric, not the paraphrase below.
        # Mixing the two puts arms on different scales: on the same generations the
        # paraphrase scored ~0.4 lower, which is larger than most arm differences.
        from judge_giants import JUDGE_PROMPT
        return JUDGE_PROMPT.format(gold=gold.strip(), pred=pred.strip())
    obj, gold_tag, pred_tag = HEAD[task]
    return (
        f"Below is a {obj}:\n<{gold_tag}>\n{gold.strip()}\n</{gold_tag}>\n"
        f"Below is a statement you need to evaluate:\n<{pred_tag}>\n{pred.strip()}\n</{pred_tag}>\n"
        + RUBRIC_TAIL.format(obj=obj, axes=AXES[task])
    )


def extract_pred(task: str, text: str):
    if task == "giants":
        return extract_insight(text)
    return extract_tagged(text, {"abgen": "design", "hypoarena": "hypothesis",
                                 "prescience": "paper"}[task])


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(HEAD))
    p.add_argument("--gen", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--max-judge-tokens", type=int, default=4096)
    a = p.parse_args()

    local = "openrouter.ai" not in a.base_url
    key = os.environ.get(a.api_key_env) or os.environ.get(a.api_key_env + "_NEW") or ("EMPTY" if local else None)
    if not key:
        sys.exit(f"set ${a.api_key_env}")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=a.base_url, api_key=key, timeout=1800.0, max_retries=0)
    rows = [json.loads(l) for l in open(a.gen) if l.strip()]
    seen = {}
    for r in rows:
        seen[r["id"]] = r
    rows = list(seen.values())

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"[judge:{a.task}] {a.judge} rows={len(rows)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    fh = open(a.out, "a")
    t0 = time.time()
    cnt = {"n": 0, "err": 0, "empty": 0}

    async def one(r):
        pred = extract_pred(a.task, r.get("output") or "")
        gold = (r.get("meta") or {}).get("gold") or ""
        if not pred or not gold:
            async with lock:
                cnt["empty"] += 1
                fh.write(json.dumps({"id": r["id"], "rating": None, "reason": "no_insight",
                                     "meta": r.get("meta")}, ensure_ascii=False) + "\n")
                fh.flush()
            return
        prompt = build_prompt(a.task, gold, pred)
        last = None
        for k in range(a.retries + 1):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=a.judge,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=a.max_judge_tokens,
                    )
                txt = resp.choices[0].message.content or ""
                rating = parse_rating(txt)
                if rating is None and k < a.retries:
                    last = ValueError("unparseable rating")
                    await asyncio.sleep(2 * (k + 1))
                    continue
                async with lock:
                    cnt["n"] += 1
                    fh.write(json.dumps({"id": r["id"], "rating": rating, "judge": a.judge,
                                         "raw": txt[-1200:], "meta": r.get("meta")},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                    if cnt["n"] % 25 == 0:
                        print(f"[judge:{a.task}] {cnt['n']}/{len(todo)} err={cnt['err']} "
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
    print(f"[judge:{a.task}] DONE n={cnt['n']} err={cnt['err']} empty={cnt['empty']} "
          f"wall={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
