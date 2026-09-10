"""gen_client.py -- run the generation-side research-taste tasks against a vLLM endpoint.

Companion to idea_client.py. That one grades a letter against a key; this one keeps
what the model actually wrote, because two of the four tasks here have no key and are
settled later by judge_pairwise.py.

Sampling matches EVAL_PLAN.md (temperature 1.0, top_p 0.95, top_k 20, thinking on) so
these sit next to the FrontierCS / ALE / research numbers without an asterisk. The
default max_tokens is 32768 -- the length the models were *trained* at. The earlier
8192 default silently turned AAAR into a test of who finishes thinking soonest, and the
24576 re-run flipped the ranking outright (base9b_v2c 0.4617 -> 0.6200,
rlv5_ft03nm_a20_s20 0.5367 -> 0.5917), so never lower it without saying so out loud.

Two kinds of item:
  kind=gen      free-form prose. Stored verbatim in `text`; no score here.
  kind=numeric  a 0-10 rating ending in "ANSWER: <n>". Parsed to `pred`, compared to
                `target` offline by genstats.py via Spearman -- see gen_prep.py for
                why rank correlation and not absolute error.

  python3 gen_client.py --tasks gentasks.jsonl --out DIR --base-url URL --model TAG
"""
import hashlib, os, argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests

NUM_RE = re.compile(r"ANSWER\s*:\s*(-?\d+(?:\.\d+)?)", re.I)


def parse_number(text):
    """last ANSWER: wins; fall back to a bare number on one of the final lines."""
    if not text:
        return None
    body = text.split("</think>")[-1]
    hits = NUM_RE.findall(body) or NUM_RE.findall(text)
    if hits:
        try:
            v = float(hits[-1])
        except ValueError:
            return None
        return min(10.0, max(0.0, v))
    for line in reversed([l.strip() for l in body.splitlines() if l.strip()][-3:]):
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(?:/\s*10)?", line)
        if m:
            return min(10.0, max(0.0, float(m.group(1))))
    return None


def _seed(item, k):
    """Legacy: seed = 1000*k + 7 for every item (one shared random stream per sample index).
    GEN_SEED_HASH=1: per-(task,id,k) seed so draws are independent across items."""
    if os.environ.get("GEN_SEED_HASH", "0") == "1":
        h = hashlib.sha1(f"{item['task']}|{item['id']}".encode()).hexdigest()
        return (int(h[:8], 16) + k) % (2 ** 31)
    return 1000 * k + 7


def one(args, item, k):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": args.temperature, "top_p": args.top_p,
        "max_tokens": args.max_tokens, "n": 1, "seed": _seed(item, k),
        "extra_body": {"top_k": args.top_k},
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
            rec = {"task": item["task"], "id": item["id"], "sample_idx": k,
                   "kind": item.get("kind", "gen"),
                   "completion_tokens": j.get("usage", {}).get("completion_tokens"),
                   "finish_reason": ch.get("finish_reason"),
                   "seconds": round(time.time() - t0, 2),
                   "think_chars": len(think)}
            if item.get("kind") == "numeric":
                p = parse_number(txt)
                rec.update({"pred": p, "target": item.get("target"),
                            "unparsed": p is None, "text_tail": txt[-400:]})
            else:
                # the judge reads this, so keep the visible answer whole and drop the
                # chain of thought -- judging the reasoning would grade a different thing
                rec.update({"text": txt.split("</think>")[-1].strip()[:8000],
                            "unparsed": not txt.strip()})
            return rec
        except Exception as e:
            if attempt == 3:
                return {"task": item["task"], "id": item["id"], "sample_idx": k,
                        "kind": item.get("kind", "gen"), "unparsed": True,
                        "target": item.get("target"),
                        "error": f"{type(e).__name__}: {e}"[:300],
                        "seconds": round(time.time() - t0, 2)}
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--timeout", type=float, default=2400)
    ap.add_argument("--only-task", default=None)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.tasks)]
    if a.only_task:
        items = [i for i in items if i["task"] == a.only_task]
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/samples.jsonl"

    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("error"):
                done.add((r["task"], r["id"], r["sample_idx"]))
        print(f"[gen] resuming, {len(done)} usable rows already on disk")

    work = [(it, k) for it in items for k in range(a.n_samples)
            if (it["task"], it["id"], k) not in done]
    print(f"[gen] {len(items)} items x {a.n_samples} = {len(items)*a.n_samples} draws, "
          f"{len(work)} to run, concurrency {a.concurrency}", flush=True)

    t0, n = time.time(), 0
    with open(path, "a") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for res in ex.map(lambda w: one(a, w[0], w[1]), work):
            f.write(json.dumps(res) + "\n")
            n += 1
            if n % 50 == 0:
                f.flush()
                print(f"[gen] {n}/{len(work)}  {time.time()-t0:.0f}s", flush=True)

    summarise(path, f"{a.out}/summary.json")


def spearman(xs, ys):
    """rank correlation, average ranks for ties. No scipy in this venv."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


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
        kind = next(iter(next(iter(items.values())).values())).get("kind", "gen")
        if kind == "numeric":
            xs, ys, unp, n_ok = [], [], [], 0
            for iid, v in items.items():
                preds = [v[k]["pred"] for k in sorted(v) if v[k].get("pred") is not None]
                unp.append(1.0 - len(preds) / max(1, len(v)))
                if not preds:
                    continue
                xs.append(sum(preds) / len(preds))          # mean of the draws
                ys.append(v[next(iter(v))]["target"])
                n_ok += 1
            summary[task] = {"n_items": n_ok, "kind": "numeric",
                             "spearman_mean_of_draws": spearman(xs, ys),
                             "pred_mean": (sum(xs) / len(xs)) if xs else None,
                             "pred_std": (sum((x - sum(xs) / len(xs)) ** 2 for x in xs)
                                          / len(xs)) ** 0.5 if len(xs) > 1 else None,
                             "unparsed_rate": sum(unp) / len(unp) if unp else None}
        else:
            lens, unp, n_ok = [], [], 0
            for iid, v in items.items():
                ts = [v[k].get("text", "") for k in sorted(v)]
                unp.append(sum(1 for t in ts if not t) / max(1, len(ts)))
                lens += [len(t) for t in ts if t]
                n_ok += 1
            summary[task] = {"n_items": n_ok, "kind": "gen",
                             "mean_chars": (sum(lens) / len(lens)) if lens else None,
                             "empty_rate": sum(unp) / len(unp) if unp else None}
    json.dump(summary, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
