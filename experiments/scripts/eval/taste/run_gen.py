#!/usr/bin/env python3
"""Generation client for the taste-eval suite (vLLM OpenAI-compatible server).

One process = one (model, benchmark) arm.  Appends to `--out` and resumes by id,
so it can be re-run after a crash without redoing finished work.

    python run_gen.py --bench scijudge --data <jsonl> --base-url http://127.0.0.1:PORT/v1 \
        --model <served-name> --out out.jsonl --concurrency 48
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benches  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, choices=sorted(benches.LOADERS))
    p.add_argument("--data", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=None, help="subsample size (bench-specific)")
    p.add_argument("--limit", type=int, default=None, help="hard cap on tasks (debug)")
    p.add_argument("--max-tokens", type=int, default=16384)
    # Qwen3.5 model card, thinking mode / general tasks:
    #   temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
    #   presence_penalty=1.5, repetition_penalty=1.0
    # This is also the project's own "采样协议（不可动）" (EVAL_ON_JIAOLAB_zh.md 4.2).
    # presence_penalty=1.5 is what stops these models spending 32k tokens in a
    # 25-gram loop; leaving it at 0 (as the first version of this harness did)
    # turns a taste benchmark into a termination benchmark.
    # repetition_penalty MUST stay 1.0 -- 1.15 collapsed FrontierCS 7.231 -> 0.666
    # on the 9B wd0.3 arm (model card of frontiersmith-q35-9b-rl-soupNEW10-step20).
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--profile", choices=("thinking", "instruct", "raw"), default="thinking",
                   help="Qwen3.5 card presets: thinking = T1.0/top_p0.95 (general tasks), "
                        "instruct = T0.7/top_p0.8 (non-thinking general tasks). Both use "
                        "presence_penalty 1.5 / top_k 20 / min_p 0 / repetition_penalty 1.0. "
                        "raw = use whatever the individual flags say.")
    p.add_argument("--thinking", choices=("on", "off"), default="on",
                   help="off sends chat_template_kwargs.enable_thinking=false, which makes "
                        "the Qwen3.5 template emit a closed empty <think> block; use it for "
                        "the no-thinking arm of a judgement benchmark")
    return p.parse_args()


async def main():
    a = parse_args()
    if a.profile == "thinking":
        a.temperature, a.top_p = 1.0, 0.95
    elif a.profile == "instruct":
        a.temperature, a.top_p = 0.7, 0.8
    from openai import AsyncOpenAI

    tasks = benches.LOADERS[a.bench](a.data, n=a.n, seed=a.seed)
    if a.limit:
        tasks = tasks[: a.limit]

    done: set[str] = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [t for t in tasks if t["id"] not in done]
    print(
        f"[gen] bench={a.bench} model={a.model} tasks={len(tasks)} done={len(done)} todo={len(todo)}\n"
        f"[gen] sampling: T={a.temperature} top_p={a.top_p} top_k={a.top_k} min_p={a.min_p} "
        f"presence_penalty={a.presence_penalty} repetition_penalty={a.repetition_penalty} "
        f"max_tokens={a.max_tokens} thinking={a.thinking}",
        flush=True,
    )
    if not todo:
        return

    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", timeout=a.timeout, max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    fh = open(a.out, "a")
    t0 = time.time()
    state = {"n": 0, "err": 0, "trunc": 0, "ctok": 0}

    async def one(task):
        msgs = []
        if task["system"]:
            msgs.append({"role": "system", "content": task["system"]})
        msgs.append({"role": "user", "content": task["user"]})
        last = None
        for attempt in range(a.retries + 1):
            try:
                async with sem:
                    r = await client.chat.completions.create(
                        model=a.model,
                        messages=msgs,
                        max_tokens=a.max_tokens,
                        temperature=a.temperature,
                        top_p=a.top_p,
                        presence_penalty=a.presence_penalty,
                        seed=a.seed,
                        extra_body={
                            "top_k": a.top_k,
                            "min_p": a.min_p,
                            "repetition_penalty": a.repetition_penalty,
                            "chat_template_kwargs": {"enable_thinking": a.thinking == "on"},
                        },
                    )
                ch = r.choices[0]
                rec = {
                    "id": task["id"],
                    "output": ch.message.content or "",
                    "finish_reason": ch.finish_reason,
                    "prompt_tokens": r.usage.prompt_tokens if r.usage else None,
                    "completion_tokens": r.usage.completion_tokens if r.usage else None,
                    "meta": task["meta"],
                }
                async with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    state["n"] += 1
                    state["ctok"] += rec["completion_tokens"] or 0
                    if ch.finish_reason == "length":
                        state["trunc"] += 1
                    if state["n"] % 50 == 0:
                        el = time.time() - t0
                        print(
                            f"[gen] {state['n']}/{len(todo)} err={state['err']} "
                            f"trunc={state['trunc']} avg_ctok={state['ctok']/max(1,state['n']):.0f} "
                            f"{el/60:.1f}min",
                            flush=True,
                        )
                return
            except Exception as e:  # noqa: BLE001
                last = e
                await asyncio.sleep(min(30, 2 ** attempt))
        async with lock:
            state["err"] += 1
            fh.write(
                json.dumps(
                    {"id": task["id"], "output": "", "finish_reason": "error",
                     "error": f"{type(last).__name__}: {last}", "meta": task["meta"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()

    await asyncio.gather(*(one(t) for t in todo))
    fh.close()
    el = time.time() - t0
    print(
        f"[gen] DONE {state['n']}/{len(todo)} err={state['err']} trunc={state['trunc']} "
        f"avg_ctok={state['ctok']/max(1,state['n']):.0f} wall={el/60:.1f}min",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
