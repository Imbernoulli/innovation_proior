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

--v2 FIXES THREE SAMPLING DEFECTS AN ADVERSARIAL REVIEW FOUND IN THE FIRST BUILD.
The default path is left exactly as it was so the numbers already reported stay
reproducible; --v2 is a separate task file, not an edit of the old one.

  aaar_equation      v1 presented options_list in its stored order, and the stored
                     order is not uniform -- "always answer the modal position" scored
                     30.8%, not 25%. v2 shuffles the four candidates per item and
                     remaps the answer letter, restoring a true 25% floor.
  openreview_pair    v1 drew i and j with rng.randrange on every attempt, i.e. WITH
                     replacement across attempts: 4 of 120 pairs were exact duplicates
                     and 25 papers appeared in more than one pair. Correlated items
                     break the item-level bootstrap, which assumes items are
                     exchangeable. v2 draws each paper at most once.
  openreview_decide  v1 put the venue in the prompt ("submitted to ICLR 2023"). Venue
                     acceptance rates differ a lot, so a model can score by recalling
                     the venue's base rate without reading the paper. That shortcut is
                     open to every arm equally, so it does not bias the RANKING -- but
                     it means the task may not measure research judgement at all, and
                     openreview_decide is the one task where our RL arms beat the
                     control. v2 removes the venue. The 50/50 balance is unchanged, so
                     the chance floor stays 50%.

  liveidea_pair is NOT built in v2: its labels average originality over every idea
  sharing a (keyword, idea_model) key while the prompt shows only the first idea, and
  47133 of 48131 groups contain more than one distinct text. It needs a label rebuild,
  not a resample.

--venue-ablation IS THE CONTROLLED VERSION OF THE openreview_decide FIX.
--v2 shuffles AAAR's options from the same rng that later feeds the OpenReview
builders, so the rng state diverges and --v2 draws a DIFFERENT 120 papers for
openreview_decide -- zero overlap with the v1 set. That makes "score dropped after
removing the venue" uninterpretable: removing the venue and swapping the papers are
confounded. (aaar_equation is unaffected: its 120 items are chosen before any option
shuffle, so v1 and v2 share all 120 and that contrast IS clean.)

--venue-ablation sidesteps the rng entirely: it reads an existing task file and
re-emits its openreview_decide items with the venue sentence replaced, keeping the id,
the label and the paper. Same papers, same order, one word changed -- so the
difference is the venue and nothing else.

  python3 idea_prep.py [--out FILE] [--n-per-task 120] [--seed 0] [--v2]
  python3 idea_prep.py --venue-ablation --from-tasks tasks.jsonl --out tasks_v2b.jsonl
"""
import argparse, json, glob, random, os, re

D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
IB = f"{D}/ideabench"

# how much of the surrounding paper to show for the equation task. The full LaTeX
# source runs to hundreds of KB; the equation's own neighbourhood is what carries the
# signal, and a 41668-token serve context puts a hard ceiling on it anyway.
CTX_BEFORE = 6000
CTX_AFTER = 2000


def aaar_equation(n, rng, v2=False):
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
        if v2:
            # v1 showed options_list in its stored order and that order is not uniform,
            # so the majority position scored 30.8% instead of 25%. Shuffle, then find
            # where the true answer landed.
            true_opt = opts[letters.index(ans)]
            opts = list(opts)
            rng.shuffle(opts)
            ans = letters[opts.index(true_opt)]
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


def openreview_pair(df, n, rng, min_gap=0.25, v2=False):
    d = df.dropna(subset=["mean_score", "abstract", "title", "venue"])
    by_venue = {}
    for v, g in d.groupby("venue"):
        if len(g) >= 2:
            by_venue[v] = g.reset_index(drop=True)
    venues = sorted(by_venue)
    used = set()          # v2 only: (venue, row index) already spent on some pair
    out, tries = [], 0
    while len(out) < n and tries < n * 400:
        tries += 1
        v = venues[rng.randrange(len(venues))]
        g = by_venue[v]
        i, j = rng.randrange(len(g)), rng.randrange(len(g))
        if i == j:
            continue
        if v2 and ((v, i) in used or (v, j) in used):
            continue     # each paper appears in at most one pair, so items stay independent
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
        if v2:
            used.add((v, i)); used.add((v, j))
        out.append({"task": "openreview_pair", "id": f"orp{len(out)}", "prompt": prompt,
                    "answer": "A" if hi_is_a else "B", "choices": ["A", "B"],
                    "meta": {"venue": str(a.venue),
                             "gap": abs(float(a.mean_score) - float(b.mean_score))}})
    return out


def openreview_decide(df, n, rng, v2=False):
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
        if v2:
            # No venue: the model has to read the paper instead of recalling a base rate.
            lead = ("The following paper was submitted to a machine-learning conference. "
                    "Decide whether the programme committee accepted or rejected it.")
        else:
            lead = (f"The following paper was submitted to {r.venue}. Decide whether the "
                    "programme committee accepted or rejected it.")
        prompt = (
            f"{lead}\n\n"
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


VENUE_LEAD = ("The following paper was submitted to a machine-learning conference. "
              "Decide whether the programme committee accepted or rejected it.")


def venue_ablation(src):
    """Re-emit an existing file's openreview_decide items with the venue removed.

    Paired by construction: same papers, same labels, same ids as the source file, so
    the only thing that differs between the two runs is the venue sentence.
    """
    out = []
    for line in open(src):
        r = json.loads(line)
        if r.get("task") != "openreview_decide":
            continue
        body = r["prompt"].split("\n\n", 1)[1]      # drop the venue-bearing lead only
        r = dict(r, prompt=f"{VENUE_LEAD}\n\n{body}")
        r.setdefault("meta", {})["venue_removed"] = True
        out.append(r)
    if not out:
        raise SystemExit(f"no openreview_decide items in {src}")
    return out


# ---------------------------------------------------------------------------
# v3: research JUDGEMENT against outcomes, not against reviewers.
#
# WHY THIS EXISTS. openreview_score / _pair / _decide all ask the model to predict
# what the REVIEWERS said. Measured on this very corpus, within a single venue and
# restricted to accepted papers, reviewer mean_score correlates with the paper's
# eventual citation rate at Spearman +0.05 to +0.18. So those tasks have a target
# that is nearly uncorrelated with research value: a model with perfect judgement
# would still score near chance on them. That is a target-validity failure, not a
# power failure, and no amount of extra items fixes it.
#
# These tasks use the OUTCOME as ground truth instead: log citations per month,
# compared only within the same venue (same field norms, same citation window) and
# only among ACCEPTED papers (so the accept/reject signal cannot leak in).
#
# impact_contrarian is the point of the whole exercise. It keeps only the pairs
# where the reviewers preferred the paper that went on to be cited LESS. A model
# that has merely learned to imitate reviewer taste scores BELOW 50% there, while a
# model with independent judgement scores above it. No other task here separates
# those two hypotheses.
#
# CONFOUND, stated up front: "which paper was cited more" is partly answerable by
# recognising a famous paper rather than by judging it. meta.year is recorded on
# every item precisely so the result can be split by publication year -- an effect
# that lives only in older, more-memorisable papers is memory, not judgement.
MIN_MONTHS = 12          # a paper needs a citation window before its rate means anything
MIN_LC_GAP = 1.0         # |delta log1p(citations/month)|; ~e-fold, well clear of noise
MIN_REV_FLIP = 0.15      # how wrong the reviewers must be to count as contrarian


def _impact_frame():
    import pandas as pd, numpy as np
    df = _openreview()
    d = df[df.decision == True].dropna(                       # noqa: E712
        subset=["avg_citations_per_month", "mean_score", "abstract", "title", "venue"])
    d = d[d.month_since_publication.astype(float) >= MIN_MONTHS]
    return d.assign(lc=np.log1p(d.avg_citations_per_month.astype(float)),
                    ms=d.mean_score.astype(float)).reset_index(drop=True)


def _year(venue):
    m = re.search(r"(20\d\d)", str(venue))
    return int(m.group(1)) if m else None


def _impact_pairs(d, n, rng, contrarian):
    """Same venue, both accepted, big citation gap; optionally reviewer-inverted."""
    import numpy as np
    by_venue = {v: g.reset_index(drop=True) for v, g in d.groupby("venue") if len(g) >= 2}
    venues = sorted(by_venue)
    used, out, tries = set(), [], 0
    while len(out) < n and tries < n * 3000:
        tries += 1
        v = venues[rng.randrange(len(venues))]
        g = by_venue[v]
        i, j = rng.randrange(len(g)), rng.randrange(len(g))
        if i == j or (v, i) in used or (v, j) in used:
            continue                      # each paper used once -> items stay independent
        a, b = g.iloc[i], g.iloc[j]
        dl = float(a.lc) - float(b.lc)
        if abs(dl) < MIN_LC_GAP:
            continue
        hi, lo = (a, b) if dl > 0 else (b, a)
        rev_edge = float(hi.ms) - float(lo.ms)     # >0: reviewers agreed with posterity
        if contrarian and rev_edge > -MIN_REV_FLIP:
            continue
        if not contrarian and rev_edge < MIN_REV_FLIP:
            continue
        hi_is_a = rng.random() < 0.5               # randomise which side the winner sits on
        pa, pb = (hi, lo) if hi_is_a else (lo, hi)
        task = "impact_contrarian" if contrarian else "impact_pair"
        prompt = (
            "Two papers were accepted at the same machine-learning conference. Judge "
            "which one turned out to matter more to the field.\n\n"
            f"=== PAPER A ===\nTitle: {pa.title}\nAbstract: {pa.abstract}\n\n"
            f"=== PAPER B ===\nTitle: {pb.title}\nAbstract: {pb.abstract}\n\n"
            "Which paper went on to be cited more per month since publication? Answer "
            "on the last line in the form: ANSWER: A  or  ANSWER: B"
        )
        used.add((v, i)); used.add((v, j))
        out.append({"task": task, "id": f"{'ic' if contrarian else 'ip'}{len(out)}",
                    "prompt": prompt, "answer": "A" if hi_is_a else "B",
                    "choices": ["A", "B"],
                    "meta": {"venue": str(v), "year": _year(v),
                             "lc_gap": abs(dl), "reviewer_edge": rev_edge,
                             "hi_cites": float(hi.avg_citations_per_month),
                             "lo_cites": float(lo.avg_citations_per_month)}})
    return out


def novelty_pair(n, rng):
    """Pairwise novelty instead of pointwise: pointwise predictions collapsed onto
    7.0/7.5/8.0 for every arm, which caps the achievable Spearman by ties alone."""
    df = _openreview()
    d = df[df.decision == True].dropna(                       # noqa: E712
        subset=["mean_novelty", "abstract", "title", "venue"]).reset_index(drop=True)
    by_venue = {v: g.reset_index(drop=True) for v, g in d.groupby("venue") if len(g) >= 2}
    venues = sorted(by_venue)
    used, out, tries = set(), [], 0
    while len(out) < n and tries < n * 3000:
        tries += 1
        v = venues[rng.randrange(len(venues))]
        g = by_venue[v]
        i, j = rng.randrange(len(g)), rng.randrange(len(g))
        if i == j or (v, i) in used or (v, j) in used:
            continue
        a, b = g.iloc[i], g.iloc[j]
        gap = float(a.mean_novelty) - float(b.mean_novelty)
        if abs(gap) < 0.25:
            continue
        hi_is_a = rng.random() < 0.5
        pa, pb = (a, b) if (gap > 0) == hi_is_a else (b, a)
        prompt = (
            "Two papers were accepted at the same machine-learning conference.\n\n"
            f"=== PAPER A ===\nTitle: {pa.title}\nAbstract: {pa.abstract}\n\n"
            f"=== PAPER B ===\nTitle: {pb.title}\nAbstract: {pb.abstract}\n\n"
            "Which paper is more original -- further from what the field was already "
            "doing? Answer on the last line in the form: ANSWER: A  or  ANSWER: B"
        )
        used.add((v, i)); used.add((v, j))
        out.append({"task": "novelty_pair", "id": f"nv{len(out)}", "prompt": prompt,
                    "answer": "A" if hi_is_a else "B", "choices": ["A", "B"],
                    "meta": {"venue": str(v), "year": _year(v), "gap": abs(gap)}})
    return out


def order_swap(rows):
    """Emit every pair twice, A/B and B/A.

    Measured on openreview_pair: every one of the 8 arms is 13.3 to 20.7 points more
    accurate when the correct paper happens to sit in slot A. Randomising the side
    (which these builders already do) does not remove that -- it just makes half the
    items systematically harder and spends the power on a coin flip. Asking both
    orders converts that dead mass into signal, doubles the item count with no new
    ground truth, and yields a per-arm order-consistency rate for free: an arm that
    answers "A" both times has told us it is reading the position, not the paper.

    Items keep meta.pair_id so the two orders can be re-joined, and meta.order marks
    which is which.
    """
    MA, MB = "=== PAPER A ===\n", "=== PAPER B ===\n"
    out = []
    for r in rows:
        head, rest = r["prompt"].split(MA, 1)
        a_body, b_body = rest.split(MB, 1)
        # b_body still carries the trailing question; keep it attached to slot B so
        # the two orders are byte-identical apart from which paper sits where.
        tail_at = b_body.rindex("\n\nWhich ")
        b_text, tail = b_body[:tail_at], b_body[tail_at:]
        a_text = a_body.rstrip("\n")
        swapped = f"{head}{MA}{b_text.rstrip()}\n\n{MB}{a_text}{tail}"
        first = dict(r, id=f"{r['id']}o1",
                     meta=dict(r["meta"], pair_id=r["id"], order=1))
        second = dict(r, id=f"{r['id']}o2", prompt=swapped,
                      answer=("B" if r["answer"] == "A" else "A"),
                      meta=dict(r["meta"], pair_id=r["id"], order=2))
        out.append(first)
        out.append(second)
    return out


def judgement_suite(n, seed):
    """Each task gets its OWN rng. Sharing one rng across task builders is what
    silently re-sampled openreview_decide onto a different 120 papers when the AAAR
    option shuffle was added -- the confound that voided that whole comparison."""
    d = _impact_frame()
    rows = (_impact_pairs(d, n, random.Random(seed + 101), contrarian=False)
            + _impact_pairs(d, n, random.Random(seed + 202), contrarian=True)
            + novelty_pair(max(n * 3 // 5, 1), random.Random(seed + 303)))
    return order_swap(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{IB}/tasks.jsonl")
    ap.add_argument("--n-per-task", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v2", action="store_true",
                    help="apply the three sampling fixes described in the header")
    ap.add_argument("--venue-ablation", action="store_true",
                    help="re-emit --from-tasks' openreview_decide items without the venue")
    ap.add_argument("--judgement", action="store_true",
                    help="v3 research-judgement suite scored against citation outcomes "
                         "rather than against reviewer scores (see header above)")
    ap.add_argument("--from-tasks", default=f"{IB}/tasks.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    if a.judgement:
        rows = judgement_suite(a.n_per_task, a.seed)
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        from collections import Counter
        c = Counter(r["task"] for r in rows)
        print(f"wrote {a.out}: {len(rows)} items  {dict(c)}")
        for t in sorted(c):
            sub = [r for r in rows if r["task"] == t]
            base = Counter(r["answer"] for r in sub)
            print(f"  {t:20s} n={len(sub):4d}  majority-class={max(base.values())/len(sub):.1%}  "
                  f"median chars={sorted(len(r['prompt']) for r in sub)[len(sub)//2]}  "
                  f"years={sorted({r['meta'].get('year') for r in sub})}")
        return

    if a.venue_ablation:
        rows = venue_ablation(a.from_tasks)
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {a.out}: {len(rows)} venue-stripped openreview_decide items "
              f"paired 1:1 with {a.from_tasks}")
        return

    rows = aaar_equation(a.n_per_task, rng, v2=a.v2)
    df = _openreview()
    rows += openreview_pair(df, a.n_per_task, rng, v2=a.v2)
    rows += openreview_decide(df, a.n_per_task, rng, v2=a.v2)
    if not a.v2:
        rows += liveidea_pair(a.n_per_task, rng)   # voided in v2, see header

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
