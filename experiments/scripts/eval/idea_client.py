"""idea_client.py -- run the research-taste task file against a vLLM endpoint.

Sampling matches the main campaign (EVAL_PLAN.md): temperature 1.0, top_p 0.95,
top_k 20, N=5 per item, thinking on. Same protocol means these numbers sit next to
the FrontierCS / ALE / research numbers without an asterisk, and it gives the same
@5 family of statistics -- mean@5 is the accuracy you get from one draw, pass@5 is
whether any of five draws found it, worst@5 is whether all five did.

Every task is scored by exact match on a letter or a word, so there is no judge and
no partial credit. A response that never emits a parseable ANSWER counts as wrong,
not as missing -- refusing to answer is a failure of the task, not of the harness --
but it is tracked separately as `unparsed` so a model that degenerates is visible
rather than silently scoring 25%.

  python3 idea_client.py --tasks tasks.jsonl --out DIR --base-url URL --model TAG
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests

ANS_RE = re.compile(r"ANSWER\s*:\s*([A-Za-z]+)", re.I)


def parse_answer(text, choices):
    """last ANSWER: wins; fall back to a bare final-line choice."""
    if not text:
        return None
    body = text.split("</think>")[-1]
    hits = ANS_RE.findall(body) or ANS_RE.findall(text)
    up = {c.upper(): c for c in choices}
    for h in reversed(hits):
        h = h.strip().upper()
        if h in up:
            return up[h]
        if len(h) == 1 and h in up:
            return up[h]
    for line in reversed([l.strip() for l in body.splitlines() if l.strip()][-3:]):
        t = re.sub(r"[^A-Za-z]", "", line).upper()
        if t in up:
            return up[t]
    return None


def one(args, item, k):
    # IDEA_SYSTEM_PROMPT: optional system message prepended to every request, mirroring
    # gen_client.py's GEN_SYSTEM_PROMPT. Unset/empty = historical behaviour (bare user prompt),
    # so every number already on disk is unchanged; the unified 2026 protocol sets it to
    # "It is now year 2026. You are a good researcher." -- these tasks had NO system prompt at all
    # before, which is the one place the year could silently go missing.
    msgs = [{"role": "user", "content": item["prompt"]}]
    sysp = os.environ.get("IDEA_SYSTEM_PROMPT", "").strip()
    if sysp:
        msgs = [{"role": "system", "content": sysp}] + msgs
    payload = {
        "model": args.model,
        "messages": msgs,
        "temperature": args.temperature, "top_p": args.top_p,
        # See gen_client.py: presence_penalty=1.5 aligns this client with the RL training
        # rollout and the main bench; min_p/repetition_penalty are sent explicitly, not defaulted.
        "presence_penalty": args.presence_penalty,
        "max_tokens": args.max_tokens, "n": 1, "seed": 1000 * k + 7,
        "extra_body": {"top_k": args.top_k, "min_p": args.min_p,
                       "repetition_penalty": args.repetition_penalty},
    }
    t0 = time.time()
    for attempt in range(4):
        try:
            r = requests.post(f"{args.base_url}/chat/completions", json=payload,
                              timeout=args.timeout)
            r.raise_for_status()
            j = r.json()
            ch = j["choices"][0]
            txt = ch["message"].get("content") or ""
            think = ch["message"].get("reasoning_content") or ""
            got = parse_answer(txt, item["choices"])
            return {"task": item["task"], "id": item["id"], "sample_idx": k,
                    "answer": item["answer"], "pred": got,
                    "correct": bool(got == item["answer"]),
                    "unparsed": got is None,
                    "completion_tokens": j.get("usage", {}).get("completion_tokens"),
                    "seconds": round(time.time() - t0, 2),
                    "text_tail": txt[-600:], "think_chars": len(think),
                    "meta": item.get("meta")}
        except Exception as e:
            if attempt == 3:
                return {"task": item["task"], "id": item["id"], "sample_idx": k,
                        "answer": item["answer"], "pred": None, "correct": False,
                        "unparsed": True, "error": f"{type(e).__name__}: {e}"[:300],
                        "seconds": round(time.time() - t0, 2)}
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--only-task", default=None)
    a = ap.parse_args()
    # Auditable in the job log: the sampling protocol this run actually sent. Must match the RL
    # training rollout (temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5,
    # repetition_penalty 1.0, max 32768) or the arm is being measured off-protocol.
    print("[idea-client] sampling: " + repr({
        "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "min_p": a.min_p,
        "presence_penalty": a.presence_penalty, "repetition_penalty": a.repetition_penalty,
        "max_tokens": a.max_tokens, "n_samples": a.n_samples}), flush=True)

    items = [json.loads(l) for l in open(a.tasks)]
    if a.only_task:
        items = [i for i in items if i["task"] == a.only_task]
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/samples.jsonl"

    done = set()
    if os.path.exists(path):                       # resume: same contract as the main client
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("error"):
                done.add((r["task"], r["id"], r["sample_idx"]))
        print(f"[idea] resuming, {len(done)} usable rows already on disk")

    work = [(it, k) for it in items for k in range(a.n_samples)
            if (it["task"], it["id"], k) not in done]
    print(f"[idea] {len(items)} items x {a.n_samples} = {len(items)*a.n_samples} draws, "
          f"{len(work)} to run, concurrency {a.concurrency}", flush=True)

    t0, n = time.time(), 0
    with open(path, "a") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for res in ex.map(lambda w: one(a, w[0], w[1]), work):
            f.write(json.dumps(res) + "\n")
            n += 1
            if n % 50 == 0:
                f.flush()
                print(f"[idea] {n}/{len(work)}  {time.time()-t0:.0f}s", flush=True)

    summarise(path, f"{a.out}/summary.json")


def summarise(samples_path, out_path):
    from collections import defaultdict
    per = defaultdict(lambda: defaultdict(dict))
    for line in open(samples_path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        per[r["task"]][r["id"]][r["sample_idx"]] = r
    summary = {}
    for task, items in per.items():
        full = [v for v in items.values() if len(v) >= 5]
        if not full:
            continue
        mean5, pass5, all5, unp = [], [], [], []
        for v in full:
            c = [v[k]["correct"] for k in sorted(v)[:5]]
            u = [v[k].get("unparsed", False) for k in sorted(v)[:5]]
            mean5.append(sum(c) / len(c))
            pass5.append(1.0 if any(c) else 0.0)
            all5.append(1.0 if all(c) else 0.0)
            unp.append(sum(u) / len(u))
        summary[task] = {
            "n_items": len(full),
            "mean@5": sum(mean5) / len(mean5),
            "pass@5": sum(pass5) / len(pass5),
            "all5": sum(all5) / len(all5),
            "unparsed_rate": sum(unp) / len(unp),
        }
    json.dump(summary, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))


def _log_sysprompt():
    _sp = os.environ.get("IDEA_SYSTEM_PROMPT", "").strip()
    print("[idea-client] system message: "
          + (repr({"role": "system", "content": _sp}) if _sp else "(none)"), flush=True)

if __name__ == "__main__":
    _log_sysprompt()
    main()
