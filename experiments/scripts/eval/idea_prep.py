"""idea_prep.py -- build the "research taste / research judge" task file.

The four benchmarks below all measure judgement rather than execution: nothing here
runs an experiment, the model only has to tell good work from bad. They are all
third-party data with ground truth already in it, and every one is scored by exact
match against that ground truth, so no LLM judge sits in the loop -- which matters,
because a judge drawn from the same family as the models under test would quietly
reward self-similarity.

  aaar_equation      Reza8848/AAAR-1.0, Equation Inference. 4-way multiple choice:
                     one real equation from a paper against three perturbations that
                     are syntactically valid LaTeX. Ground truth = the published one.
  openreview_pair    nhop/OpenReview. Two papers from the SAME venue, which one drew
                     the higher mean review score. Pairwise on purpose: it cancels
                     out whatever absolute scale a model invents for itself, so a
                     harsh model and a generous model are compared on ordering only.
  openreview_decide  Same source, accept vs reject, balanced 50/50 by construction
                     (the raw pool is 65/35, and an unbalanced pool lets "always
                     accept" look like taste).
  liveidea_pair      6cf/liveideabench-v2. Two ideas answering the SAME keyword,
                     which one the critics found more original. Ground truth is the
                     CONSENSUS across every critic model that scored both ideas, not
                     one critic's opinion.

Pairs are only kept when the ground-truth gap is wide enough to be a real preference
rather than tie-breaking noise, and A/B order is randomised so "always answer A"
scores 50%.

  python3 idea_prep.py [--out FILE] [--n-per-task 120] [--seed 0]
"""
import argparse, json, glob, random, os, re

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
IB = f"{D}/ideabench"

# how much of the surrounding paper to show for the equation task. The full LaTeX
# source runs to hundreds of KB; the equation's own neighbourhood is what carries the
# signal, and a 41668-token serve context puts a hard ceiling on it anyway.
CTX_BEFORE = 6000
CTX_AFTER = 2000


def aaar_equation(n, rng):
    p = f"{IB}/aaar/Equation_Inference/equation.1049.json"
    rows = json.load(open(p))
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        opts = r.get("options_list") or []
        ans = (r.get("answer") or "").strip().upper()
        if len(opts) != 4 or ans not in "ABCD":
            continue
        before = (r.get("context_before") or "")[-CTX_BEFORE:]
        after = (r.get("context_after") or "")[:CTX_AFTER]
        letters = "ABCD"
        body = "\n".join(f"({letters[j]}) {o}" for j, o in enumerate(opts))
        prompt = (
            "You are reading the LaTeX source of a machine-learning paper. One equation "
            "has been removed and replaced by [MASK]. Exactly one of the four candidates "
            "below is the equation the authors actually wrote; the other three are "
            "plausible but wrong.\n\n"
            f"--- context before [MASK] ---\n{before}\n\n"
            f"--- context after [MASK] ---\n{after}\n\n"
            f"--- candidates ---\n{body}\n\n"
            "Which candidate is the real equation? Answer with the single letter A, B, C "
            "or D on the last line, in the form: ANSWER: <letter>"
        )
        out.append({"task": "aaar_equation", "id": f"eq{i}", "prompt": prompt,
                    "answer": ans, "choices": list(letters)})
        if len(out) >= n:
            break
    return out


def _openreview():
    import pandas as pd
    files = sorted(glob.glob(f"{IB}/openreview/data/*.parquet"))
    if not files:
        raise SystemExit(f"no OpenReview parquet under {IB}/openreview/data")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def openreview_pair(df, n, rng, min_gap=0.25):
    d = df.dropna(subset=["mean_score", "abstract", "title", "venue"])
    by_venue = {}
    for v, g in d.groupby("venue"):
        if len(g) >= 2:
            by_venue[v] = g.reset_index(drop=True)
    venues = sorted(by_venue)
    out, tries = [], 0
    while len(out) < n and tries < n * 400:
        tries += 1
        g = by_venue[venues[rng.randrange(len(venues))]]
        i, j = rng.randrange(len(g)), rng.randrange(len(g))
        if i == j:
            continue
        a, b = g.iloc[i], g.iloc[j]
        if abs(float(a.mean_score) - float(b.mean_score)) < min_gap:
            continue
        hi_is_a = float(a.mean_score) > float(b.mean_score)
        if rng.random() < 0.5:            # randomise which side the winner sits on
            a, b, hi_is_a = b, a, not hi_is_a
        prompt = (
            "Two papers were submitted to the same venue. Judge which one the reviewers "
            "rated higher overall.\n\n"
            f"=== PAPER A ===\nTitle: {a.title}\nAbstract: {a.abstract}\n\n"
            f"=== PAPER B ===\nTitle: {b.title}\nAbstract: {b.abstract}\n\n"
            "Which paper received the higher average review score? Answer on the last "
            "line in the form: ANSWER: A  or  ANSWER: B"
        )
        out.append({"task": "openreview_pair", "id": f"orp{len(out)}", "prompt": prompt,
                    "answer": "A" if hi_is_a else "B", "choices": ["A", "B"],
                    "meta": {"venue": str(a.venue),
                             "gap": abs(float(a.mean_score) - float(b.mean_score))}})
    return out


def openreview_decide(df, n, rng):
    d = df.dropna(subset=["decision", "abstract", "title"])
    acc = d[d.decision == True].reset_index(drop=True)      # noqa: E712
    rej = d[d.decision == False].reset_index(drop=True)     # noqa: E712
    half = n // 2
    idx_a = rng.sample(range(len(acc)), min(half, len(acc)))
    idx_r = rng.sample(range(len(rej)), min(n - half, len(rej)))
    rows = [(acc.iloc[i], "ACCEPT") for i in idx_a] + [(rej.iloc[i], "REJECT") for i in idx_r]
    rng.shuffle(rows)
    out = []
    for k, (r, lab) in enumerate(rows):
        prompt = (
            f"The following paper was submitted to {r.venue}. Decide whether the "
            "programme committee accepted or rejected it.\n\n"
            f"Title: {r.title}\nAbstract: {r.abstract}\n\n"
            "Answer on the last line in the form: ANSWER: ACCEPT  or  ANSWER: REJECT"
        )
        out.append({"task": "openreview_decide", "id": f"ord{k}", "prompt": prompt,
                    "answer": lab, "choices": ["ACCEPT", "REJECT"],
                    "meta": {"venue": str(r.venue)}})
    return out


def liveidea_pair(n, rng, min_gap=1.5, max_words=400):
    import pandas as pd
    p = f"{IB}/liveidea/liveideabench_hf.csv"
    df = pd.read_csv(p, usecols=["keywords", "idea_model", "critic_model", "idea",
                                 "originality", "idea_length_in_words"])
    df = df.dropna(subset=["idea", "originality", "keywords"])
    df = df[df.idea_length_in_words <= max_words]
    # consensus: average every critic's originality for the same (keyword, idea_model)
    cons = (df.groupby(["keywords", "idea_model"])
              .agg(orig=("originality", "mean"), n_crit=("originality", "size"),
                   idea=("idea", "first")).reset_index())
    cons = cons[cons.n_crit >= 3]
    out = []
    for kw, g in cons.groupby("keywords"):
        if len(g) < 2:
            continue
        g = g.reset_index(drop=True)
        for _ in range(3):
            i, j = rng.randrange(len(g)), rng.randrange(len(g))
            if i == j:
                continue
            a, b = g.iloc[i], g.iloc[j]
            if abs(a.orig - b.orig) < min_gap:
                continue
            hi_is_a = a.orig > b.orig
            if rng.random() < 0.5:
                a, b, hi_is_a = b, a, not hi_is_a
            prompt = (
                f'Two research ideas were proposed for the topic "{kw}". Judge which one '
                "is more original -- which makes the more novel scientific contribution, "
                "as opposed to restating what the field already does.\n\n"
                f"=== IDEA A ===\n{a.idea}\n\n=== IDEA B ===\n{b.idea}\n\n"
                "Which idea is more original? Answer on the last line in the form: "
                "ANSWER: A  or  ANSWER: B"
            )
            out.append({"task": "liveidea_pair", "id": f"lip{len(out)}", "prompt": prompt,
                        "answer": "A" if hi_is_a else "B", "choices": ["A", "B"],
                        "meta": {"keyword": str(kw), "gap": float(abs(a.orig - b.orig))}})
            break
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{IB}/tasks.jsonl")
    ap.add_argument("--n-per-task", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = aaar_equation(a.n_per_task, rng)
    df = _openreview()
    rows += openreview_pair(df, a.n_per_task, rng)
    rows += openreview_decide(df, a.n_per_task, rng)
    rows += liveidea_pair(a.n_per_task, rng)

    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter(r["task"] for r in rows)
    print(f"wrote {a.out}: {len(rows)} items  {dict(c)}")
    for t in c:
        sub = [r for r in rows if r["task"] == t]
        base = Counter(r["answer"] for r in sub)
        chance = max(base.values()) / len(sub)
        print(f"  {t:20s} n={len(sub):4d}  majority-class baseline={chance:.1%}  "
              f"median prompt chars={sorted(len(r['prompt']) for r in sub)[len(sub)//2]}")


if __name__ == "__main__":
    main()
