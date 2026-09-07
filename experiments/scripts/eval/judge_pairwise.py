"""judge_pairwise.py -- blind pairwise LLM judging of two arms' generated output.

WHAT THIS ANSWERS
The multiple-choice tasks ask whether a model can *recognise* good research. This asks
whether it can *produce* it, which is what the SFT and RL data were actually about.

THE JUDGE, AND WHY IT IS ALLOWED TO BE OUR OWN BASE MODEL
The judge is the un-finetuned Qwen the whole campaign descends from. That sounds like
it should self-favour, and for an absolute score it would. It does not bias *this*
measurement, because every comparison is our-arm vs the control-arm at the SAME stage,
and both sides are equidistant descendants of the judge: our-SFT vs base, our-RL vs
base-RL. A judge that likes its own idiom likes it in both candidates. What a weak
judge costs us is power, not direction -- it makes wins harder to see, not fake.

THREE BIASES THAT ARE ACTUALLY CONTROLLED
  position   every pair is judged twice with the candidates swapped. A win counts only
             when both orders agree. Disagreements become ties, and the agreement rate
             is reported -- it is the honest measure of how much signal the judge has.
             A judge at 50% agreement is a coin and its "win rate" means nothing.
  length     the rubric tells the judge to ignore length and polish, and we log the
             character counts anyway so a length confound is visible after the fact.
  grounding  on review_weakness the judge is not asked for its taste at all. It is
             shown the REAL reviews and asked which candidate matches a concern the
             actual reviewers raised. That converts a judgment call into a
             comparison against ground truth.

  python3 judge_pairwise.py --a-dir OUT_ours --b-dir OUT_control --tasks gentasks.jsonl \
      --out DIR --base-url URL --model judge
"""
import argparse, json, os, random, re, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

VERDICT_RE = re.compile(r"VERDICT\s*:\s*([AB])\b", re.I)

IDEA_RUBRIC = """You are a senior researcher comparing two research ideas written for the same keyword.

Keyword: {keyword}

[Candidate A]
{a}

[Candidate B]
{b}

Judge only the science, in this priority order:
1. Originality -- a genuinely new angle beats a restatement of standard practice.
2. Specificity -- an idea a researcher could act on beats a vague direction.
3. Soundness -- the reasoning must be technically correct.
4. Testability -- a concrete falsifying experiment beats a gesture at evaluation.

Ignore length, formatting and writing polish entirely. A shorter, plainer idea that is
more original and more specific must win.

Reply with a two-sentence justification, then exactly one final line:
VERDICT: A
or
VERDICT: B"""

WEAKNESS_RUBRIC = """Two reviewers each named what they consider the single most important weakness of the same paper. Your job is to decide which one better matches the concerns the paper's ACTUAL reviewers raised.

Paper title: {title}

Excerpts from the real peer reviews this paper received:
{reviews}

[Candidate A]
{a}

[Candidate B]
{b}

Which candidate identifies a weakness that the real reviewers actually raised, stated specifically and technically correctly? A candidate that names a real, substantive concern beats one that is generic ("needs more experiments", "writing could be clearer") even if the generic one is better written. Ignore length and polish.

Reply with a two-sentence justification, then exactly one final line:
VERDICT: A
or
VERDICT: B"""


def load_gen(d):
    """task -> id -> sample_idx -> text"""
    out = defaultdict(lambda: defaultdict(dict))
    p = f"{d}/samples.jsonl"
    if not os.path.exists(p):
        sys.exit(f"no samples.jsonl in {d}")
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error") or r.get("kind") != "gen":
            continue
        t = (r.get("text") or "").strip()
        if t:
            out[r["task"]][r["id"]][r["sample_idx"]] = t
    return out


def build_prompt(task, meta, a_text, b_text):
    if task == "liveidea_gen":
        return IDEA_RUBRIC.format(keyword=meta.get("keyword", ""), a=a_text, b=b_text)
    revs = meta.get("real_reviews") or []
    joined = "\n\n".join(f"--- reviewer {i+1} ---\n{r[:1800]}" for i, r in enumerate(revs[:3]))
    return WEAKNESS_RUBRIC.format(title=meta.get("title", ""), reviews=joined,
                                  a=a_text, b=b_text)


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
            hits = VERDICT_RE.findall(body) or VERDICT_RE.findall(txt)
            return (hits[-1].upper() if hits else None), txt[-300:]
        except Exception as e:
            if attempt == 3:
                return None, f"ERR {type(e).__name__}: {e}"[:200]
            time.sleep(3 * (attempt + 1))


def one_pair(args, job):
    """Judge one (item, sample) pair in BOTH orders; agreement is the whole point."""
    task, iid, k, meta, a_text, b_text = job
    # order 1: ours is A. order 2: ours is B. A consistent judge must flip its letter.
    v1, tail1 = ask(args, build_prompt(task, meta, a_text, b_text))
    v2, tail2 = ask(args, build_prompt(task, meta, b_text, a_text))
    if v1 is None or v2 is None:
        outcome = "unparsed"
    elif v1 == "A" and v2 == "B":
        outcome = "a_wins"
    elif v1 == "B" and v2 == "A":
        outcome = "b_wins"
    else:
        outcome = "inconsistent"
    return {"task": task, "id": iid, "sample_idx": k, "outcome": outcome,
            "v_order1": v1, "v_order2": v2,
            "a_chars": len(a_text), "b_chars": len(b_text),
            "tail1": tail1, "tail2": tail2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True, help="OUR arm's gen output dir")
    ap.add_argument("--b-dir", required=True, help="CONTROL arm's gen output dir")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--a-tag", default="A")
    ap.add_argument("--b-tag", default="B")
    ap.add_argument("--n-pairs", type=int, default=4, help="samples per item to judge")
    ap.add_argument("--only-task", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.3)
    # The judge is a thinking model, so this budget binds exactly the way the eval
    # budget did. At 4096 it never reached a VERDICT on 15-17% of review_weakness
    # calls -- long prompt, long deliberation, cut off mid-sentence. liveidea_gen was
    # unaffected (0% unparsed), which is why only the long task needed a re-run.
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=float, default=1200)
    a = ap.parse_args()

    meta = {}
    for line in open(a.tasks):
        it = json.loads(line)
        if it.get("kind") == "gen":
            meta[(it["task"], it["id"])] = it

    A, B = load_gen(a.a_dir), load_gen(a.b_dir)
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/judgements.jsonl"

    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("outcome") != "unparsed":
                done.add((r["task"], r["id"], r["sample_idx"]))
        print(f"[judge] resuming, {len(done)} pairs already judged")

    jobs = []
    for task in sorted(set(A) & set(B)):
        if a.only_task and task != a.only_task:
            continue
        for iid in sorted(set(A[task]) & set(B[task])):
            for k in range(a.n_pairs):
                if k in A[task][iid] and k in B[task][iid] \
                        and (task, iid, k) not in done and (task, iid) in meta:
                    jobs.append((task, iid, k, meta[(task, iid)],
                                 A[task][iid][k], B[task][iid][k]))
    random.Random(7).shuffle(jobs)   # spread tasks across the run so partials are usable
    print(f"[judge] {a.a_tag} vs {a.b_tag}: {len(jobs)} pairs x 2 orders "
          f"= {2*len(jobs)} calls, concurrency {a.concurrency}", flush=True)

    t0, n = time.time(), 0
    with open(path, "a") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for res in ex.map(lambda j: one_pair(a, j), jobs):
            f.write(json.dumps(res) + "\n")
            n += 1
            if n % 25 == 0:
                f.flush()
                print(f"[judge] {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    summarise(path, f"{a.out}/summary.json", a.a_tag, a.b_tag)


def summarise(path, out_path, a_tag, b_tag):
    per = defaultdict(list)
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        per[r["task"]].append(r)

    summary = {"a": a_tag, "b": b_tag, "tasks": {}}
    for task, rows in sorted(per.items()):
        aw = sum(1 for r in rows if r["outcome"] == "a_wins")
        bw = sum(1 for r in rows if r["outcome"] == "b_wins")
        inc = sum(1 for r in rows if r["outcome"] == "inconsistent")
        unp = sum(1 for r in rows if r["outcome"] == "unparsed")
        dec = aw + bw
        # item-level win rate, so 4 correlated draws of one item are not 4 data points
        byitem = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["outcome"] == "a_wins":
                byitem[r["id"]][0] += 1
            elif r["outcome"] == "b_wins":
                byitem[r["id"]][1] += 1
        item_scores = [w / (w + l) for w, l in byitem.values() if w + l]
        summary["tasks"][task] = {
            "n_pairs": len(rows),
            "a_wins": aw, "b_wins": bw,
            "inconsistent": inc, "unparsed": unp,
            "decided_rate": dec / len(rows) if rows else None,
            "win_rate_pairs": aw / dec if dec else None,
            "n_items": len(item_scores),
            "win_rate_items": (sum(item_scores) / len(item_scores)) if item_scores else None,
            "a_mean_chars": sum(r["a_chars"] for r in rows) / len(rows) if rows else None,
            "b_mean_chars": sum(r["b_chars"] for r in rows) / len(rows) if rows else None,
        }
    json.dump(summary, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
