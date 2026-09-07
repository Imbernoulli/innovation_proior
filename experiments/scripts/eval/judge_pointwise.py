"""judge_pointwise.py -- score each candidate ALONE against ground truth.

WHY THIS EXISTS
judge_pairwise.py showed both candidates at once and asked which was better. On
liveidea_gen that works (0% unparsed, ~58-62% of pairs decided consistently across the
two orders). On review_weakness it collapsed: only ~22% of pairs got a consistent
verdict, and among the calls that did answer, the judge named the SECOND-shown
candidate about 87% of the time in BOTH orders.

CAVEAT ON THAT DIAGNOSIS -- it is confounded, and the confound is mine. The judge ran
at max_tokens=4096 and never reached a VERDICT on 15-17% of review_weakness calls, all
cut off mid-deliberation. So "severe position bias" and "the judge was truncated on the
hardest items and guessed" are not yet separated. Both judges now default to 16384 and
review_weakness is being re-run; until that lands, treat the position-bias number as
provisional rather than established.

Showing one candidate at a time removes the position failure mode by construction --
there is no "second position" to prefer. The cost is that absolute scores are less
sensitive than a direct comparison, so this is the right tool only where pairwise
demonstrably breaks.

The comparison is still paired: both arms answer the same items, and the bootstrap
resamples items, scoring both arms on each draw (see genstats.py for why).

  python3 judge_pointwise.py --a-dir OUT_ours --b-dir OUT_control --tasks gentasks.jsonl \
      --out DIR --base-url URL --model judge --task review_weakness
"""
import argparse, json, os, random, re, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

SCORE_RE = re.compile(r"SCORE\s*:\s*([0-5])\b", re.I)

WEAKNESS_RUBRIC = """A reviewer named what they consider the single most important weakness of the paper below. Your job is to score how well that matches the concerns the paper's ACTUAL reviewers raised.

Paper title: {title}

Excerpts from the real peer reviews this paper received:
{reviews}

The candidate weakness to score:
{cand}

Score 0 to 5:
5 = names a specific, substantive concern that the real reviewers clearly raised
4 = names a real concern the reviewers raised, but less precisely
3 = plausibly related to a real concern, stated vaguely
2 = generic reviewer boilerplate ("needs more experiments", "unclear writing") that happens to overlap
1 = generic boilerplate unrelated to what the reviewers said
0 = wrong, incoherent, or about a different paper

Judge only the substance. Ignore length, fluency and formatting -- a blunt one-sentence answer that names the right concern scores above a polished paragraph that does not.

Reply with one sentence of justification, then exactly one final line:
SCORE: <0-5>"""

IDEA_RUBRIC = """Score the research idea below, written in response to a keyword.

Keyword: {title}

The idea:
{cand}

Score 0 to 5 on originality and specificity combined:
5 = a genuinely new angle, concrete enough to act on tomorrow, with a falsifying test
4 = novel and specific, but the test is vague
3 = a reasonable idea that mostly recombines standard practice
2 = a restatement of what the field already does
1 = vague direction with no actionable content
0 = off-topic or incoherent

Ignore length, fluency and formatting entirely.

Reply with one sentence of justification, then exactly one final line:
SCORE: <0-5>"""


def load_gen(d, task):
    out = defaultdict(dict)
    p = f"{d}/samples.jsonl"
    if not os.path.exists(p):
        sys.exit(f"no samples.jsonl in {d}")
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error") or r.get("kind") != "gen" or r.get("task") != task:
            continue
        t = (r.get("text") or "").strip()
        if t:
            out[r["id"]][r["sample_idx"]] = t
    return out


def build(task, meta, cand):
    if task == "liveidea_gen":
        return IDEA_RUBRIC.format(title=meta.get("keyword", ""), cand=cand)
    revs = meta.get("real_reviews") or []
    joined = "\n\n".join(f"--- reviewer {i+1} ---\n{r[:1800]}" for i, r in enumerate(revs[:3]))
    return WEAKNESS_RUBRIC.format(title=meta.get("title", ""), reviews=joined, cand=cand)


def ask(args, prompt):
    payload = {"model": args.model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": args.temperature, "top_p": 0.95,
               "max_tokens": args.max_tokens, "n": 1,
               "extra_body": {"top_k": 20}}
    for attempt in range(4):
        try:
            r = requests.post(f"{args.base_url}/chat/completions", json=payload,
                              timeout=args.timeout)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"].get("content") or ""
            body = txt.split("</think>")[-1]
            hits = SCORE_RE.findall(body) or SCORE_RE.findall(txt)
            return (int(hits[-1]) if hits else None), txt[-200:]
        except Exception as e:
            if attempt == 3:
                return None, f"ERR {type(e).__name__}: {e}"[:200]
            time.sleep(3 * (attempt + 1))


def one(args, job):
    task, iid, k, side, meta, cand = job
    s, tail = ask(args, build(task, meta, cand))
    return {"task": task, "id": iid, "sample_idx": k, "side": side,
            "score": s, "chars": len(cand), "tail": tail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True)
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="review_weakness")
    ap.add_argument("--a-tag", default="A")
    ap.add_argument("--b-tag", default="B")
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.3)
    # Thinking-model budget: at 3072 this judge failed to emit a SCORE on 66% of
    # review_weakness calls (315/480), all cut off mid-deliberation, which left the
    # comparison resting on a third of the items. Same class of error as the 8192 eval
    # budget -- a binding budget silently changes what is being measured.
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=float, default=1200)
    a = ap.parse_args()

    meta = {}
    for line in open(a.tasks):
        it = json.loads(line)
        if it.get("kind") == "gen" and it["task"] == a.task:
            meta[it["id"]] = it

    A, B = load_gen(a.a_dir, a.task), load_gen(a.b_dir, a.task)
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/scores.jsonl"

    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("score") is not None:
                done.add((r["id"], r["sample_idx"], r["side"]))
        print(f"[pw] resuming, {len(done)} scores on disk")

    jobs = []
    for iid in sorted(set(A) & set(B) & set(meta)):
        for k in range(a.n_pairs):
            if k in A[iid] and k in B[iid]:
                if (iid, k, "a") not in done:
                    jobs.append((a.task, iid, k, "a", meta[iid], A[iid][k]))
                if (iid, k, "b") not in done:
                    jobs.append((a.task, iid, k, "b", meta[iid], B[iid][k]))
    random.Random(7).shuffle(jobs)
    print(f"[pw] {a.a_tag} vs {a.b_tag} on {a.task}: {len(jobs)} single-candidate scorings",
          flush=True)

    t0, n = time.time(), 0
    with open(path, "a") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for res in ex.map(lambda j: one(a, j), jobs):
            f.write(json.dumps(res) + "\n")
            n += 1
            if n % 50 == 0:
                f.flush()
                print(f"[pw] {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    summarise(path, f"{a.out}/summary.json", a.a_tag, a.b_tag, a.task)


def summarise(path, out_path, a_tag, b_tag, task):
    per = defaultdict(lambda: {"a": [], "b": []})
    unp = 0
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("score") is None:
            unp += 1
            continue
        per[r["id"]][r["side"]].append(r["score"])

    items = [(sum(v["a"]) / len(v["a"]), sum(v["b"]) / len(v["b"]))
             for v in per.values() if v["a"] and v["b"]]
    if not items:
        json.dump({"error": "no scored items"}, open(out_path, "w"))
        return
    d = [x - y for x, y in items]
    mean = sum(d) / len(d)
    rng = random.Random(20260907)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    summary = {
        "a": a_tag, "b": b_tag, "task": task, "n_items": n,
        "a_mean_score": sum(x for x, _ in items) / n,
        "b_mean_score": sum(y for _, y in items) / n,
        "diff": mean,
        "ci95": [bs[250], bs[9750]],
        "p_gt0": sum(1 for x in bs if x > 0) / len(bs),
        "wins": sum(1 for x in d if x > 0), "losses": sum(1 for x in d if x < 0),
        "ties": sum(1 for x in d if x == 0),
        "unparsed": unp,
    }
    json.dump(summary, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
