#!/usr/bin/env python3
"""gen_prep.py -- build the generation-side research-taste tasks.

The four tasks in ideabench/tasks.jsonl are all multiple choice: the model picks a
letter and we grade it against a key. That measures discrimination, and it turned out
to measure something else too -- at max_tokens=8192 the AAAR ranking was driven by who
ran out of budget before writing "ANSWER:", not by who was right (base9b_v2c goes
0.4617 -> 0.6200 when the budget is lifted to 24576). Discrimination tasks also cannot
say anything about the thing we actually trained for, which is *producing* research.

So this file builds the other half: tasks where the model writes something, and the
grade comes either from a real number the reviewers wrote down, or from a blind judge.

  liveidea_gen      LiveIdeaBench keyword -> a scientific idea. No key exists, so this
                    is scored by pairwise LLM judging (judge_pairwise.py).
  review_weakness   paper abstract -> the single most important weakness. Judged
                    pairwise, but the judge is *grounded*: it sees what the real
                    reviewers wrote and is asked which candidate matches a concern
                    they actually raised. That keeps the judge from grading on prose.
  openreview_score  paper -> predicted overall rating. Fully objective: Spearman
                    against the real reviewer mean. No judge in the loop at all.
  openreview_novel  paper -> predicted novelty rating. Same, against mean_novelty.
                    This is the closest thing in the data to "research taste".

WHY SPEARMAN AND NOT MAE
mean_score in this dump is min-max normalised to [0,1] per the source dataset, and the
normalisation is not comparable across venues (ICLR 2024 spans 0.0-0.9, CoRL 2024
0.3-1.0). Absolute error would therefore mostly measure whether a model guessed the
right scale. Rank correlation is invariant to that, so we ask for a 0-10 rating,
never calibrate it, and only ever compare orderings.

CONTAMINATION, STATED UP FRONT
These are 2018-2024 submissions and the models may well have memorised some outcomes.
That inflates every arm's correlation. It does not bias the *comparison*, because all
arms share one base and see identical items -- but no absolute number here should be
read as "the model can predict peer review".
"""
import argparse, glob, json, os, random, re, sys

import pandas as pd

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
IB = f"{D}/ideabench"

LIVEIDEA_PROMPT = """You are an experienced scientist. Propose one novel research idea connected to the keyword below.

Keyword: {keyword}

Your idea must be specific enough that another researcher could act on it tomorrow. Cover, in this order:
1. The precise gap or failure in current work.
2. Your proposed approach, concretely.
3. Why it is genuinely novel rather than an increment.
4. The experiment that would falsify it.

Write at most 300 words. Do not pad or restate the keyword."""

WEAKNESS_PROMPT = """Below is a paper submitted to a machine learning conference.

Title: {title}

Abstract: {abstract}

You are reviewing this submission. Identify the SINGLE most important weakness that would determine whether it is accepted. Be specific and technical -- name the assumption, the missing experiment, or the unsupported claim. Do not list several weaknesses, and do not summarise the paper.

Write at most 150 words."""

SCORE_PROMPT = """Below is a paper submitted to {venue_h}.

Title: {title}

Abstract: {abstract}

Reviewers scored this submission for {dim_h}. Predict the average score the reviewers gave, on a 0 to 10 scale where 0 is the weakest submission at this venue and 10 the strongest. Only the ordering of your scores matters, so use the full range rather than clustering everything near the middle.

Think it through, then end your reply with exactly one line:
ANSWER: <number between 0 and 10>"""

DIM_H = {
    "openreview_score": "overall quality (the recommendation score)",
    "openreview_novel": "novelty and originality specifically",
}
DIM_COL = {"openreview_score": "mean_score", "openreview_novel": "mean_novelty"}


def venue_human(v):
    m = re.match(r"^([A-Za-z]+)[_.]?.*?(\d{4})", str(v))
    if m:
        return f"{m.group(1).upper()} {m.group(2)}"
    return "a machine learning conference"


def load_openreview():
    parts = sorted(glob.glob(f"{IB}/openreview/data/*.parquet"))
    if not parts:
        sys.exit("no openreview parquet; run idea_prep.py first")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def clean_abstract(a, cap=2200):
    a = re.sub(r"\s+", " ", str(a)).strip()
    return a[:cap]


def real_weaknesses(reviews, cap=5):
    """Pull the reviewer-written prose we will later hand the judge as ground truth."""
    out = []
    if reviews is None:
        return out
    try:
        it = list(reviews)
    except TypeError:
        return out
    for r in it[:cap]:
        if not isinstance(r, dict):
            continue
        rv = r.get("review") or {}
        if not isinstance(rv, dict):
            continue
        for k in ("main_review", "limitations", "paper_summary"):
            t = rv.get(k)
            if not t:
                continue
            t = re.sub(r"^\s*%s:\s*" % k, "", re.sub(r"\s+", " ", str(t))).strip()
            # "see above" and friends carry no signal for the judge
            if len(t) > 200:
                out.append(t[:2500])
                break
    return out


def build_liveidea(n, rng):
    csv = f"{IB}/liveidea/liveideabench_hf.csv"
    df = pd.read_csv(csv, usecols=["keywords"])
    # keywords that many idea_models were run on are the benchmark's real prompts;
    # singletons are usually artefacts of one model's run.
    counts = df["keywords"].value_counts()
    pool = sorted(counts[counts >= 20].index.tolist())
    rng.shuffle(pool)
    return [
        {"task": "liveidea_gen", "id": f"liveidea_gen/{i}", "keyword": kw,
         "prompt": LIVEIDEA_PROMPT.format(keyword=kw), "kind": "gen"}
        for i, kw in enumerate(pool[:n])
    ]


def build_weakness(orv, n, rng):
    df = orv[orv["reviews"].notna()].copy()
    df["_w"] = df["reviews"].map(real_weaknesses)
    df = df[df["_w"].map(len) >= 2]
    df = df[df["abstract"].astype(str).str.len().between(600, 4000)]
    idx = list(df.index)
    rng.shuffle(idx)
    out = []
    for i, ix in enumerate(idx[:n]):
        r = df.loc[ix]
        out.append({
            "task": "review_weakness", "id": f"review_weakness/{i}",
            "title": str(r["title"]).strip(),
            "prompt": WEAKNESS_PROMPT.format(title=str(r["title"]).strip(),
                                             abstract=clean_abstract(r["abstract"])),
            "kind": "gen",
            "real_reviews": r["_w"][:3],
        })
    return out


def build_numeric(orv, task, n, rng):
    col = DIM_COL[task]
    df = orv[pd.to_numeric(orv[col], errors="coerce").notna()].copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["abstract"].astype(str).str.len().between(600, 4000)]
    # Spread the draw over the score range. Picking uniformly would hand us a pile of
    # mid-scoring papers, and rank correlation on a squashed target is mostly noise.
    df["_b"] = pd.qcut(df[col], 6, labels=False, duplicates="drop")
    per = max(1, n // df["_b"].nunique())
    picks = []
    for _, g in df.groupby("_b"):
        picks += list(g.sample(min(per, len(g)), random_state=rng.randint(0, 10**6)).index)
    rng.shuffle(picks)
    out = []
    for i, ix in enumerate(picks[:n]):
        r = df.loc[ix]
        out.append({
            "task": task, "id": f"{task}/{i}",
            "prompt": SCORE_PROMPT.format(venue_h=venue_human(r["venue"]),
                                          title=str(r["title"]).strip(),
                                          abstract=clean_abstract(r["abstract"]),
                                          dim_h=DIM_H[task]),
            "kind": "numeric",
            "target": float(r[col]),
            "venue": str(r["venue"]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{IB}/gentasks.jsonl")
    ap.add_argument("--n-gen", type=int, default=60)
    ap.add_argument("--n-num", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260907)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    orv = load_openreview()
    print(f"openreview: {len(orv)} papers", file=sys.stderr)

    items = []
    items += build_liveidea(a.n_gen, rng)
    items += build_weakness(orv, a.n_gen, rng)
    items += build_numeric(orv, "openreview_score", a.n_num, rng)
    items += build_numeric(orv, "openreview_novel", a.n_num, rng)

    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    from collections import Counter
    c = Counter(i["task"] for i in items)
    for k, v in c.items():
        print(f"  {k:20s} {v}", file=sys.stderr)
    print(f"wrote {len(items)} -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
