#!/usr/bin/env python3
"""Prompt builders / answer parsers for the three taste-eval benchmarks.

Every prompt here is reproduced from the benchmark's own released artefact:

* GiantsBench   -- `query` field of giants2026/GiantsBench-test (verbatim; it is
                   the paper's Figure 10 insight-anticipation prompt already
                   rendered).  Judge rubric is Figure 12 of arXiv:2604.09793,
                   transcribed in judge_giants.py.
* SciJudgeBench -- `messages` of OpenMOSS-Team/SciJudgeBench.  The user turn is
                   reconstructed from the row metadata with a template verified
                   byte-exact on all 1,000 main-test rows, which is what makes
                   the A/B position swap faithful.
* SoundnessBench-- system + user template copied from
                   hosytuyen/SoundnessBench `rigorbench/evaluation/prompt.py`
                   (mode `direct_bucket`), experiments rendered with that repo's
                   `_format_experiments_for_eval`.

A "task" is {id, system, user, meta}.  `system=None` means: send no system turn.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict

# --------------------------------------------------------------------------- #
# GiantsBench
# --------------------------------------------------------------------------- #

def load_giants(path: str, n: int | None = 400, seed: int = 20260826) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    if n is not None and n < len(rows):
        # stratify by domain, deterministic
        by_dom: dict[str, list] = defaultdict(list)
        for r in rows:
            by_dom[r["domain"]].append(r)
        rng = random.Random(seed)
        picked: list = []
        doms = sorted(by_dom)
        quota = {d: max(1, round(n * len(by_dom[d]) / len(rows))) for d in doms}
        for d in doms:
            pool = sorted(by_dom[d], key=lambda r: r["pair_id"])
            rng.shuffle(pool)
            picked.extend(pool[: quota[d]])
        picked.sort(key=lambda r: r["pair_id"])
        rng.shuffle(picked)
        rows = picked[:n]
    out = []
    for r in rows:
        out.append(
            {
                "id": str(r["pair_id"]),
                "system": None,
                "user": r["query"],
                "meta": {
                    "domain": r["domain"],
                    "arxiv_id": r["arxiv_id"],
                    "gold": extract_insight(r["completion"]) or "",
                },
            }
        )
    return out


_INSIGHT_RE = re.compile(r"<insight>(.*?)</insight>", re.S | re.I)

# GiantsBench's own prompt shows this placeholder line INSIDE the <insight> tags.
# ~20% of 4B generations copy it back verbatim -- sometimes alone inside the tags
# with the real insight following after </insight>.  Handing the placeholder to
# the judge would score a good answer at the floor, so it is stripped and the
# next candidate is used instead.
_PLACEHOLDER = re.compile(
    r"^\s*A clear and self-contained statement of the insight\s*\(3-10 sentences\)\.?\s*$",
    re.I | re.M,
)
_MIN_INSIGHT_CHARS = 120


def _clean(s: str | None) -> str | None:
    if not s:
        return None
    s = _PLACEHOLDER.sub("", s).strip()
    return s or None


def extract_insight(text: str) -> str | None:
    """The model's actual insight, in order of preference.

    1. the last <insight>..</insight> block that survives placeholder-stripping
       and is long enough to be a real answer,
    2. whatever follows the last </insight> (the common "echo the placeholder,
       then write the answer" shape),
    3. whatever follows an unclosed <insight>,
    4. whatever follows the last </think>.
    """
    if not text:
        return None
    for blk in reversed(_INSIGHT_RE.findall(text)):
        c = _clean(blk)
        if c and len(c) >= _MIN_INSIGHT_CHARS:
            return c
    k = text.lower().rfind("</insight>")
    if k >= 0:
        c = _clean(text[k + len("</insight>") :])
        if c and len(c) >= _MIN_INSIGHT_CHARS:
            return c
    i = text.lower().rfind("<insight>")
    if i >= 0:
        c = _clean(text[i + len("<insight>") :])
        if c and len(c) >= _MIN_INSIGHT_CHARS:
            return c
    j = text.lower().rfind("</think>")
    if j >= 0:
        c = _clean(text[j + len("</think>") :])
        if c and len(c) >= _MIN_INSIGHT_CHARS:
            return c
    # nothing long enough: fall back to the longest short candidate rather than
    # dropping the item, so a genuinely terse answer is still judged
    cands = [_clean(b) for b in _INSIGHT_RE.findall(text)]
    cands = [c for c in cands if c]
    return max(cands, key=len) if cands else None


# --------------------------------------------------------------------------- #
# SciJudgeBench
# --------------------------------------------------------------------------- #

SCIJUDGE_SYSTEM = (
    "You are a helpful assistant. You first think about the reasoning process in "
    "your mind and then provide the user with the answer."
)

SCIJUDGE_TPL = (
    "Today is {today}. Based on the titles, abstracts, and publication dates of the "
    "following two papers A and B, determine which paper has a higher citation count.\n"
    "Show your reasoning process in <reason> </reason> tags. And return the final "
    "answer in <answer> </answer> tags. The final answer should contain only the "
    "letter A or B.\n\n"
    "Paper A (Published: {da}):\nTitle: {ta}\nAbstract: {aa}\n\n"
    "Paper B (Published: {db}):\nTitle: {tb}\nAbstract: {ab}"
)


def load_scijudge(path: str, n: int | None = None, seed: int = 20260826) -> list[dict]:
    """Two tasks per pair: `<id>` (original order) and `<id>#swap` (A<->B)."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if n is not None and n < len(rows):
        rng = random.Random(seed)
        rows = sorted(rows, key=lambda r: str(r.get("paper_a_arxiv_id")))
        rng.shuffle(rows)
        rows = rows[:n]
    tasks = []
    for i, r in enumerate(rows):
        u0 = r["messages"][1]["content"]
        today = re.match(r"Today is (\d{4}-\d{2}-\d{2})\.", u0).group(1)
        A = dict(
            d=str(r["paper_a_date"])[:10], t=r["paper_a_title"], a=r["paper_a_abstract"]
        )
        B = dict(
            d=str(r["paper_b_date"])[:10], t=r["paper_b_title"], a=r["paper_b_abstract"]
        )
        pid = r.get("paper_a_arxiv_id") or f"row{i}"
        base = f"{i:05d}_{pid}"
        gold = r["correct_answer"]
        for swap in (False, True):
            x, y = (B, A) if swap else (A, B)
            user = SCIJUDGE_TPL.format(
                today=today, da=x["d"], ta=x["t"], aa=x["a"], db=y["d"], tb=y["t"], ab=y["a"]
            )
            if swap:
                # verify we really did rebuild the released prompt for the plain order
                gold_here = {"A": "B", "B": "A"}[gold]
            else:
                assert user.strip() == u0.strip(), f"prompt reconstruction drifted at row {i}"
                gold_here = gold
            tasks.append(
                {
                    "id": base + ("#swap" if swap else ""),
                    "system": SCIJUDGE_SYSTEM,
                    "user": user,
                    "meta": {
                        "pair": base,
                        "swap": swap,
                        "gold": gold_here,
                        "category": r.get("paper_a_category"),
                        "subcategory": r.get("paper_a_subcategory"),
                    },
                }
            )
    return tasks


_ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.I)


def extract_ab(text: str) -> str | None:
    if not text:
        return None
    m = _ANSWER_RE.findall(text)
    if m:
        return m[-1].upper()
    # tolerate a missing closing tag
    m2 = re.findall(r"<answer>\s*([AB])\b", text, re.I)
    if m2:
        return m2[-1].upper()
    return None


# --------------------------------------------------------------------------- #
# SoundnessBench
# --------------------------------------------------------------------------- #

SOUNDNESS_SYSTEM = (
    "You are an expert ML researcher and peer reviewer. Classify the scientific "
    "rigor bucket of a research idea and your assessment confidence from 1 to 5 "
    "from its hypothesis and experiment description.\n\nOutput the assessment as a "
    "JSON object, including a detailed step-by-step justification for the rigor "
    "bucket selected."
)

SOUNDNESS_USER = """Classify this hypothesis-experiment pair into one rigor bucket:
- "low": "Weak scientific contribution. Hypothesis is vague or trivial, experiments lack controls or baselines, metrics are weak, or methodology has fundamental flaws.",
- "high": "Strong scientific contribution. Hypothesis is clear and meaningful. Experiments are rigorous, controlled, include appropriate baselines/ablations, and use suitable metrics.",

Confidence Score Scale:
- 1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.
- 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
- 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
- 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
- 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

HYPOTHESIS:
{hypothesis}

EXPERIMENT:
{experiment}

Output format:
{{
  "justification": "<Think step-by-step, provide detailed justification>",
  "rigor_bucket": <"low" or "high">,
  "confidence": <1-5 integer>
}}

Constraints:
- rigor_bucket must be a choice in ["low", "high"]
- confidence must be an integer in [1, 5]
"""


def _format_experiments_for_eval(experiments) -> str:
    """Verbatim from hosytuyen/SoundnessBench rigorbench/extraction/extract.py."""
    if experiments is None or len(experiments) == 0:
        return ""
    parts = []
    for i, exp in enumerate(experiments, 1):
        lines = [f"Experiment {i}:"]
        if exp.get("Description"):
            lines.append(f"  Description: {exp['Description']}")
        if exp.get("Method"):
            lines.append(f"  Method: {exp['Method']}")
        if exp.get("Evaluation Metrics"):
            lines.append(f"  Evaluation Metrics: {exp['Evaluation Metrics']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def load_soundness(path: str, n: int | None = None, seed: int = 20260826) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if n is not None and n < len(rows):
        rng = random.Random(seed)
        rows = sorted(rows, key=lambda r: r["pair_id"])
        rng.shuffle(rows)
        rows = rows[:n]
    tasks = []
    for r in rows:
        hyp = str(r.get("short_hypothesis") or r.get("hypothesis") or "").strip()
        exp = _format_experiments_for_eval(r.get("experiments") or [])
        if not exp:
            exp = str(r.get("experiment") or "").strip()
        tasks.append(
            {
                "id": str(r["pair_id"]),
                "system": SOUNDNESS_SYSTEM,
                "user": SOUNDNESS_USER.format(
                    hypothesis=hyp or "(none)", experiment=exp or "(none)"
                ),
                "meta": {
                    "gold": r["rigor_bucket"],
                    "year": r.get("year"),
                    "subfield": r.get("subfield"),
                    "soundness_score": r.get("soundness_score"),
                },
            }
        )
    return tasks


def _strip_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()


def _first_json_object(text: str) -> str | None:
    text = _strip_fences(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_bucket(text: str) -> tuple[str | None, int | None]:
    """SoundnessBench `_parse_prediction`, plus a thinking-model fallback.

    Thinking models emit the chain first and the JSON last, so we parse the
    LAST balanced object rather than the first when several are present.
    """
    if not text:
        return None, None
    body = _strip_fences(text)
    cands = []
    i = 0
    while True:
        j = body.find("{", i)
        if j < 0:
            break
        depth = 0
        for k in range(j, len(body)):
            if body[k] == "{":
                depth += 1
            elif body[k] == "}":
                depth -= 1
                if depth == 0:
                    cands.append(body[j : k + 1])
                    i = k + 1
                    break
        else:
            break
    bucket = conf = None
    for raw in reversed(cands):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            bm = re.search(r'"(?:rigor_bucket|bucket)"\s*:\s*"([^"]+)"', raw, re.I)
            cm = re.search(r'"(?:confidence|reviewer_confidence)"\s*:\s*([-+]?\d+(?:\.\d+)?)', raw)
            if bm and str(bm.group(1)).strip().lower() in ("low", "high"):
                bucket = bm.group(1).strip().lower()
                conf = _clamp_conf(cm.group(1)) if cm else None
                return bucket, conf
            continue
        if not isinstance(obj, dict):
            continue
        b = obj.get("rigor_bucket", obj.get("bucket"))
        b = str(b).strip().lower() if b is not None else None
        if b in ("low", "high"):
            return b, _clamp_conf(obj.get("confidence", obj.get("reviewer_confidence")))
    # last resort: a bare quoted verdict anywhere in the tail
    m = re.findall(r'rigor[_ ]?bucket["\s:]+"?(low|high)', body, re.I)
    if m:
        return m[-1].lower(), None
    return None, None


def _clamp_conf(v):
    try:
        return int(round(max(1.0, min(5.0, float(v)))))
    except (TypeError, ValueError):
        return None


LOADERS = {"giants": load_giants, "scijudge": load_scijudge, "soundness": load_soundness}


# --------------------------------------------------------------------------- #
# Round 2: more benchmarks
# --------------------------------------------------------------------------- #

def _swap_paper_blocks(user: str) -> str:
    """A<->B position swap done by surgery on the RELEASED prompt string.

    Re-templating cannot be used for every SciJudgeBench split (the ICLR split's
    abstracts carry trailing whitespace that a template silently normalises), so
    the two "Paper X ...:" blocks are cut out and exchanged verbatim.  The caller
    asserts that the no-op path reproduces the original byte-for-byte.
    """
    i = user.index("\n\nPaper A")
    j = user.index("\n\nPaper B")
    head, a, b = user[:i], user[i + 2 : j], user[j + 2 :]
    la, ba = a.split("\n", 1)
    lb, bb = b.split("\n", 1)
    assert head + "\n\n" + la + "\n" + ba + "\n\n" + lb + "\n" + bb == user
    new_a = lb.replace("Paper B", "Paper A", 1) + "\n" + bb
    new_b = la.replace("Paper A", "Paper B", 1) + "\n" + ba
    return head + "\n\n" + new_a + "\n\n" + new_b


def load_scijudge_iclr(path: str, n: int | None = None, seed: int = 20260826) -> list[dict]:
    """SciJudgeBench metric-OOD split: which ICLR submission is more likely accepted.

    Different template from the citation splits (no dates, `<think>` instead of
    `<reason>`), so the swap is done by block surgery on the released prompt.
    """
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if n is not None and n < len(rows):
        rng = random.Random(seed)
        rows = sorted(rows, key=lambda r: (r.get("year", 0), r["paper_a_title"]))
        rng.shuffle(rows)
        rows = rows[:n]
    tasks = []
    for i, r in enumerate(rows):
        u0 = r["messages"][1]["content"]
        base = f"{i:05d}_iclr{r.get('year')}"
        for swap in (False, True):
            user = _swap_paper_blocks(u0) if swap else u0
            gold = {"A": "B", "B": "A"}[r["correct_answer"]] if swap else r["correct_answer"]
            tasks.append(
                {
                    "id": base + ("#swap" if swap else ""),
                    "system": r["messages"][0]["content"],
                    "user": user,
                    "meta": {
                        "pair": base,
                        "swap": swap,
                        "gold": gold,
                        "category": f"ICLR{r.get('year')}",
                        "rating_a": r.get("paper_a_rating"),
                        "rating_b": r.get("paper_b_rating"),
                    },
                }
            )
    return tasks


# ---- RINoBench: novelty judgment on a 1-5 scale with the benchmark's rubric --
RINO_SYSTEM = (
    "You are an expert reviewer assessing the novelty of a research idea against "
    "the prior work that is given to you."
)

RINO_SCALE = """1: The idea is not novel. All aspects already exist in prior work.
2: The idea is marginally novel. It represents only a minor variation of existing work.
3: The idea is somewhat novel. Aspects already exist in prior work. However, it might combine known approaches in new ways, apply them to new contexts, or propose incremental updates.
4: The idea is novel. It introduces new aspects not present in existing work.
5: The idea is highly innovative and novel. It is not present in existing work and potentially encourages new thinking or opens up new research directions."""

RINO_USER = """Below is a research idea, followed by the prior work it should be judged against.

<research_idea>
Objective: {objective}

Problem statement: {problem}

Solution approach: {solution}
</research_idea>

<related_work>
{related}
</related_work>

Task: rate how novel the research idea is with respect to the related work, on this scale:

{scale}

Base the rating only on the related work shown above. Put your reasoning in <reason> </reason> tags and the final rating in <answer> </answer> tags. The answer must contain only a single integer from 1 to 5."""


def load_rino(path: str, n: int | None = None, seed: int = 20260826,
              max_works: int = 25, abstract_chars: int = 1400) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    if n is not None and n < len(rows):
        rng = random.Random(seed)
        rows = sorted(rows, key=lambda r: r["source"])
        rng.shuffle(rows)
        rows = rows[:n]
    tasks = []
    for r in rows:
        works = list(r["related_works"])[:max_works]
        rel = "\n\n".join(
            f"[{k+1}] {w['title']} ({w.get('year')})\n{(w.get('abstract') or '')[:abstract_chars]}"
            for k, w in enumerate(works)
        )
        idea = r["research_idea"]
        tasks.append(
            {
                "id": str(r["source"]).rsplit("=", 1)[-1],
                "system": RINO_SYSTEM,
                "user": RINO_USER.format(
                    objective=idea["objective"],
                    problem=idea["problem_statement"],
                    solution=idea["solution_approach"],
                    related=rel,
                    scale=RINO_SCALE,
                ),
                "meta": {
                    "gold": int(r["novelty_score"]),
                    "venue": r.get("venueid"),
                    "n_related_shown": len(works),
                    "n_related_total": len(r["related_works"]),
                },
            }
        )
    return tasks


def extract_rating_1_5(text: str):
    if not text:
        return None
    m = re.findall(r"<answer>\s*([1-5])\s*</answer>", text, re.I)
    if not m:
        m = re.findall(r"<answer>\s*([1-5])\b", text, re.I)
    if not m:
        m = re.findall(r"\b(?:rating|score)\D{0,10}\b([1-5])\b", text, re.I)
    return int(m[-1]) if m else None


# ---- AbGen: design the ablation study for a given research objective ---------
ABGEN_SYSTEM = "You are an experienced machine-learning researcher designing ablation studies."

ABGEN_USER = """<research_background>
{background}
</research_background>

<method>
{method}
</method>

<main_experiment>
{main_experiment}
</main_experiment>

Task: design the ablation study that answers this research objective:

<research_objective>
{objective}
</research_objective>

Describe the experimental setup you would run: what is varied, what is held fixed, which baselines or variants are compared, on which data, and which metrics decide the outcome. Be concrete enough that another researcher could execute it.

Put your reasoning in <think> </think> tags and the final design in <design> </design> tags."""


def load_abgen(path: str, n: int | None = 200, seed: int = 20260826) -> list[dict]:
    rows = json.load(open(path))
    if n is not None and n < len(rows):
        rng = random.Random(seed)
        rows = sorted(rows, key=lambda r: r["example_id"])
        rng.shuffle(rows)
        rows = rows[:n]
    tasks = []
    for r in rows:
        ab = r["ablation_study"]
        me = r.get("main_experiment") or {}
        tasks.append(
            {
                "id": str(r["example_id"]),
                "system": ABGEN_SYSTEM,
                "user": ABGEN_USER.format(
                    background=r["research_background"],
                    method=r["method"],
                    main_experiment=me.get("experiment_setup", ""),
                    objective=ab["research_objective"],
                ),
                "meta": {
                    "gold": ab.get("experiment_setup", ""),
                    "arxiv_id": r.get("arxiv_id"),
                    "title": r.get("title"),
                    "objective": ab["research_objective"],
                },
            }
        )
    return tasks


# ---- HypoArena: prospective hypothesis discovery from a pre-conclusion context
HYPO_SYSTEM = (
    "You are a researcher reading source material at a point where the conclusions "
    "have not yet been established."
)

HYPO_USER = """<context>
{context}
</context>

Task: standing at this point of uncertainty, state the single most important hypothesis that is worth investigating next. It must be grounded in the material above, non-trivial, and stated precisely enough that someone could design a study to support or refute it. Also say briefly what evidence would settle it.

Put your reasoning in <think> </think> tags and the final hypothesis in <hypothesis> </hypothesis> tags."""


def load_hypoarena(path: str, n: int | None = 150, seed: int = 20260826,
                   context_chars: int = 40000) -> list[dict]:
    rows = json.load(open(path))
    rows = [r for r in rows if r.get("hypotheses")]
    if n is not None and n < len(rows):
        by_dom: dict[str, list] = defaultdict(list)
        for r in rows:
            by_dom[r["domain"]].append(r)
        rng = random.Random(seed)
        picked = []
        doms = sorted(by_dom)
        per = max(1, n // len(doms))
        for d in doms:
            pool = sorted(by_dom[d], key=lambda r: r["id"])
            rng.shuffle(pool)
            picked.extend(pool[:per])
        picked.sort(key=lambda r: r["id"])
        rng.shuffle(picked)
        rows = picked[:n]
    tasks = []
    for r in rows:
        gold = "\n\n".join(
            f"{h['hypothesis']}\n\nEvidence that would settle it: {h.get('evidence','')}"
            for h in r["hypotheses"]
        )
        tasks.append(
            {
                "id": str(r["id"]),
                "system": HYPO_SYSTEM,
                "user": HYPO_USER.format(context=r["context"][:context_chars]),
                "meta": {"gold": gold, "domain": r["domain"]},
            }
        )
    return tasks


def extract_tagged(text: str, tag: str) -> str | None:
    if not text:
        return None
    m = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.S | re.I)
    if m:
        return m[-1].strip() or None
    i = text.lower().rfind(f"<{tag}>")
    if i >= 0:
        return text[i + len(tag) + 2 :].strip() or None
    j = text.lower().rfind("</think>")
    if j >= 0:
        return text[j + len("</think>") :].strip() or None
    return None


LOADERS.update(
    {
        "scijudge_iclr": load_scijudge_iclr,
        "rino": load_rino,
        "abgen": load_abgen,
        "hypoarena": load_hypoarena,
    }
)


# --------------------------------------------------------------------------- #
# Round 3: PAIR-IQ (survey §7) and SciPredict (survey §32)
# --------------------------------------------------------------------------- #

PAIRIQ_SYSTEM = (
    "You are an experienced program committee member. You first think about the "
    "reasoning process in your mind and then provide the user with the answer."
)

PAIRIQ_TPL = """Two papers were submitted to the same venue in the same year and the same primary area. Only their research idea is shown -- the goal and the approach, with no results, no author or institution information, and no title.

Based on the ideas alone, determine which submission received the higher reviewer rating.
Show your reasoning process in <reason> </reason> tags. And return the final answer in <answer> </answer> tags. The final answer should contain only the letter A or B.

Submission A:
Goal: {ma}
Approach: {aa}

Submission B:
Goal: {mb}
Approach: {ab}"""


def load_pairiq(path: str, n: int | None = 500, seed: int = 20260826,
                min_gap: float = 0.6) -> list[dict]:
    """Pairwise reviewer-rating judgement built from PAIR-IQ (survey §7 / LigBench).

    Pairs are matched on (conference, year, primary area) -- the same
    "control for field and cohort" trick SciJudgeBench uses for citations -- and
    kept only when the rating gap is >= min_gap (~1.2 sd of the rating spread),
    so the label is not noise.  Only `main` + `approach` are shown: no title, no
    venue, no results, no authors, so this measures the IDEA, not the branding.

    NB: the pairing rule is OURS; the paper's exact protocol is not published
    with the data.  Scored with the same swap-consistency rule as SciJudgeBench,
    so chance is 25%.
    """
    import glob as _glob

    files = sorted(_glob.glob(os.path.join(path, "*", "*", "*.json")))
    rows = []
    for f in files:
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if not (r.get("main") and r.get("approach")) or not isinstance(r.get("rating"), (int, float)):
            continue
        rows.append(r)
    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        buckets[(r.get("conference"), r.get("year"), r.get("primary area"))].append(r)
    rng = random.Random(seed)
    cands = []
    for key, group in sorted(buckets.items()):
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda r: r["rating"])
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                gap = group[j]["rating"] - group[i]["rating"]
                if gap >= min_gap:
                    cands.append((gap, group[i], group[j], key))
    rng.shuffle(cands)
    # spread over areas: round-robin by bucket so one big area cannot dominate
    by_key: dict[tuple, list] = defaultdict(list)
    for c in cands:
        by_key[c[3]].append(c)
    picked, keys = [], sorted(by_key)
    while keys and (n is None or len(picked) < n):
        for k in list(keys):
            if by_key[k]:
                picked.append(by_key[k].pop())
            else:
                keys.remove(k)
            if n is not None and len(picked) >= n:
                break
    tasks = []
    for i, (gap, lo, hi, key) in enumerate(picked):
        # randomise which side the winner starts on, then swap
        winner_first = rng.random() < 0.5
        X, Y = (hi, lo) if winner_first else (lo, hi)
        gold0 = "A" if winner_first else "B"
        base = f"{i:05d}_{key[0]}{key[1]}"
        for swap in (False, True):
            P, Q = (Y, X) if swap else (X, Y)
            gold = ({"A": "B", "B": "A"}[gold0]) if swap else gold0
            tasks.append(
                {
                    "id": base + ("#swap" if swap else ""),
                    "system": PAIRIQ_SYSTEM,
                    "user": PAIRIQ_TPL.format(
                        ma=P["main"], aa=P["approach"], mb=Q["main"], ab=Q["approach"]
                    ),
                    "meta": {
                        "pair": base, "swap": swap, "gold": gold,
                        "category": key[2], "venue": f"{key[0]}{key[1]}",
                        "gap": round(float(gap), 3),
                        "title_hi": hi.get("title"), "title_lo": lo.get("title"),
                    },
                }
            )
    return tasks


SCIPREDICT_SYSTEM = (
    "You are a domain expert predicting the outcome of an experiment you have not "
    "seen the results of. You first think about the reasoning process in your mind "
    "and then provide the user with the answer."
)

SCIPREDICT_TPL = """{question}

Predict the outcome. Show your reasoning process in <reason> </reason> tags, then give the letter of the single best option in <answer> </answer> tags. The final answer should contain only one letter."""


_OPT_RE = re.compile(r"^\s*\(?([A-Ea-e])[\).\:]", re.M)


def load_scipredict(path: str, n: int | None = None, seed: int = 20260826) -> list[dict]:
    """SciPredict (survey §32) -- MCQ subset only, graded against the released GTA.

    Physics / chemistry / biology experiments from papers published after
    2025-03, i.e. outside the corpus our models were built from and outside the
    base model's training window, so it doubles as a clean cross-domain OOD read.
    Free-format and numerical items are skipped: they need a judge, and the point
    of this bench in our suite is a judge-free objective column.
    """
    import pandas as pd

    df = pd.read_csv(path)
    df = df[df["PQ_FORMAT"].astype(str).str.upper().str.strip() == "MCQ"]
    tasks = []
    for i, r in df.reset_index(drop=True).iterrows():
        gta = str(r["GTA"]).strip()
        m = re.match(r"\(?([A-Ea-e])[\).\:]", gta)
        if not m:
            continue
        q = str(r["OUTCOME_PREDICTION_QUESTION"])
        if not _OPT_RE.search(q):
            continue  # options must be in the prompt or the letter is unanswerable
        tasks.append(
            {
                "id": f"{i:04d}_{str(r['DOMAIN'])[:3]}",
                "system": SCIPREDICT_SYSTEM,
                "user": SCIPREDICT_TPL.format(question=q),
                "meta": {
                    "gold": m.group(1).upper(),
                    "domain": r["DOMAIN"],
                    "field": r["FIELD"],
                    "title": r["TITLE"],
                },
            }
        )
    if n is not None and n < len(tasks):
        rng = random.Random(seed)
        rng.shuffle(tasks)
        tasks = tasks[:n]
    return tasks


def extract_letter(text: str, allowed="ABCDE") -> str | None:
    if not text:
        return None
    pat = f"[{allowed}{allowed.lower()}]"
    m = re.findall(rf"<answer>\s*\(?({pat})\b", text)
    if not m:
        m = re.findall(rf"\banswer\b\W{{0,12}}\(?({pat})[\).\s]", text, re.I)
    return m[-1].upper() if m else None


LOADERS.update({"pairiq": load_pairiq, "scipredict": load_scipredict})


# --------------------------------------------------------------------------- #
# Round 4: Lit2Test (survey §30) -- falsifiable research ideation
# --------------------------------------------------------------------------- #

LIT2TEST_FIELDS = ["literature_gap", "hypothesis", "minimal_test",
                   "decisive_metric", "supporting_result", "falsifying_result"]

LIT2TEST_SYSTEM = (
    "You are a researcher deriving the next experiment from a small literature "
    "neighborhood."
)

LIT2TEST_TPL = """{research_context}

Open problem: {open_problem}

Resource constraint: {resource_constraint}

Task: {task_instruction}

Papers in this neighborhood:
{papers}

Answer with a JSON object containing exactly these six fields:
- "literature_gap": the specific tension or gap ACROSS the supplied papers (not a generic gap that would fit any paper group)
- "hypothesis": one specific, falsifiable claim
- "minimal_test": the smallest decisive experiment, inside the resource constraint, including one baseline from a supplied paper and one failure-mode or ablation diagnostic
- "decisive_metric": the metric that settles it, matched to the mechanism
- "supporting_result": the concrete observation that would support the hypothesis
- "falsifying_result": the concrete observation that would prove it wrong

Put your reasoning in <think> </think> tags, then output the JSON object inside <proposal> </proposal> tags."""


def load_lit2test(path: str, n: int | None = None, seed: int = 20260826) -> list[dict]:
    """Lit2Test (survey §30): 200 OpenReview literature neighbourhoods.

    The six output fields and the judge prompt are the benchmark's own (the judge
    prompt is copied verbatim from its `run_lit2test_v02_pairwise.py`).  The
    GENERATION prompt is reconstructed: the repo ships the pipeline but not
    `prompts/lit2test_generation_prompt.md`, so this wording is ours while the
    inputs (`research_context` / `open_problem` / `resource_constraint` /
    `task_instruction` / the four papers) are passed through unchanged.

    Because the repo also ships the frozen six-field proposals of four frontier
    models on the SAME contexts, our arms can be judged head-to-head against
    DeepSeek-V3.2 / GLM-5 / GPT-5.2 / Sonnet-4.6 rather than only against base.
    """
    import glob as _glob

    rows = []
    for f in sorted(_glob.glob(os.path.join(path, "data", "lit2test_v02_*_contexts.jsonl"))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    if n is not None and n < len(rows):
        rng = random.Random(seed)
        rows = sorted(rows, key=lambda r: r["context_id"])
        rng.shuffle(rows)
        rows = rows[:n]
    tasks = []
    for r in rows:
        papers = "\n\n".join(
            f"[{i+1}] {p.get('title')}\n"
            f"Abstract: {(p.get('abstract') or '')[:1200]}\n"
            f"Reviewer-noted limitation: {p.get('limitation')}"
            for i, p in enumerate(r.get("papers") or [])
        )
        tasks.append(
            {
                "id": str(r["context_id"]),
                "system": LIT2TEST_SYSTEM,
                "user": LIT2TEST_TPL.format(
                    research_context=r.get("research_context", ""),
                    open_problem=r.get("open_problem", ""),
                    resource_constraint=r.get("resource_constraint", ""),
                    task_instruction=r.get("task_instruction", ""),
                    papers=papers,
                ),
                "meta": {"context_id": r["context_id"], "condition": r.get("condition"),
                         "field": r.get("field")},
            }
        )
    return tasks


def extract_proposal(text: str) -> dict | None:
    """Pull the six-field JSON out of a generation; tolerate missing tags."""
    if not text:
        return None
    body = extract_tagged(text, "proposal") or text
    body = re.sub(r"```(?:json)?", "", body)
    start = body.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(body)):
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(body[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and any(k in obj for k in LIT2TEST_FIELDS):
                        return {k: str(obj.get(k, "")).strip() for k in LIT2TEST_FIELDS}
                    break
        start = body.find("{", start + 1)
    return None


def lit2test_valid(prop: dict | None) -> bool:
    """The benchmark's own validity gate, in spirit: all six fields non-trivial."""
    if not prop:
        return False
    return all(len(prop.get(k, "")) >= 20 for k in LIT2TEST_FIELDS)


LOADERS["lit2test"] = load_lit2test


# --------------------------------------------------------------------------- #
# Round 5: PRESCIENCE contribution generation (survey §29)
# --------------------------------------------------------------------------- #

PRESCIENCE_SYSTEM = (
    "You are a researcher who has read a set of papers and is predicting the "
    "contribution of the work that builds on them."
)

PRESCIENCE_TPL = """Below are the influential references of a paper that had not yet been written when these were published.

{refs}

Predict that paper. State what it contributes -- the problem it takes on, what it does that the references do not, and what it establishes.

Put your reasoning in <think> </think> tags, then give the predicted paper as a title and an abstract inside <paper> </paper> tags, formatted as:
Title: ...
Abstract: ..."""


def load_prescience(path: str, n: int | None = 300, seed: int = 20260826,
                    min_refs: int = 3, max_refs: int = 8,
                    abstract_chars: int = 1400) -> list[dict]:
    """PRESCIENCE (survey §29): influential references -> the paper that used them.

    This is the closest shape in the whole suite to our own context.md -> answer
    format: several prior works in, one contribution out.  Targets are dated
    2024-10-01..2025-09-30, i.e. after the innovation corpus and after the base
    model's window, so contamination is structural rather than checked.

    Scored with our reference-based similarity judge, NOT the paper's LACER, so
    the numbers are internal to this run.  The paper's own headline is worth
    keeping in view when reading them: fine-tuned 7-8B models scored 4.03/3.99,
    BELOW a "hand in one of the reference papers" baseline at 4.31 -- which is
    why `--bench prescience_control` exists.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    text = {}
    for cid, ti, ab in zip(df["corpus_id"].astype(str), df["title"], df["abstract"]):
        text[cid] = (ti, ab)

    def has(roles, key):
        try:
            return key in list(roles)
        except Exception:
            return False

    tgt = df[df["roles"].apply(lambda r: has(r, "target"))]
    rows = []
    for _, t in tgt.iterrows():
        kr = t["key_references"]
        if kr is None or not hasattr(kr, "__len__"):
            continue
        refs = [str(r["corpus_id"]) for r in list(kr)]
        if not (min_refs <= len(refs) <= max_refs):
            continue
        if not all(r in text for r in refs):
            continue
        if not (t["title"] and t["abstract"]):
            continue
        rows.append((str(t["corpus_id"]), t["title"], t["abstract"], refs, str(t["date"])[:10]))
    rng = random.Random(seed)
    rows.sort(key=lambda x: x[0])
    rng.shuffle(rows)
    if n is not None:
        rows = rows[:n]
    tasks = []
    for cid, ti, ab, refs, date in rows:
        block = "\n\n".join(
            f"[{i+1}] {text[r][0]}\n{(text[r][1] or '')[:abstract_chars]}"
            for i, r in enumerate(refs)
        )
        tasks.append(
            {
                "id": cid,
                "system": PRESCIENCE_SYSTEM,
                "user": PRESCIENCE_TPL.format(refs=block),
                "meta": {
                    "gold": f"Title: {ti}\nAbstract: {ab}",
                    "date": date,
                    "n_refs": len(refs),
                    # the paper's own naive baseline: hand back one reference
                    "control": f"Title: {text[refs[0]][0]}\nAbstract: {text[refs[0]][1]}",
                },
            }
        )
    return tasks


LOADERS["prescience"] = load_prescience
