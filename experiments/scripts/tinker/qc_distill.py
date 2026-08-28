"""QC the distilled corpus against the hand-written original, turn by turn.

Three things we actually care about, in priority order:

1. VOICE. Our corpus's one distinctive asset is the first-person scientist
   ("OK, let me think this through from scratch..."). If the teacher's own
   assistant register ("The user wants me to...") replaces it, the distillation
   fixed the token distribution and threw away the thing worth keeping.

2. HINDSIGHT LEAK. The teacher was trained on (prompt -> think + answer), so when
   it regenerates the think from the prompt alone it may simply restate the answer
   it memorised. That would make the hindsight problem WORSE, not better.

   Measured as FRONT-LOADING, not total overlap. Total overlap is uninformative:
   a trace that genuinely derives the answer ends up using the answer's words, and
   much of that vocabulary was in the prompt to begin with. What separates
   derivation from restatement is WHEN the answer's prompt-absent terms appear -
   derivation introduces them progressively, restatement has them from token 0.
   On the hand-written corpus (n=2635) the median is 0.254 in the first fifth
   rising to 0.828 overall, with only 1.7% of turns above 0.5 in the first fifth:
   that is a derivation profile, so the hand-written reasoning does NOT restate
   its answer. A distilled trace that front-loads much harder is the failure we
   are watching for.

3. HEDGING. The measured defect in wd01 is that "I cannot verify" collapsed
   93.3% -> 54.4%. If the distilled reasoning hedges even less, this arm cannot fix it.
"""
import argparse, json, re, statistics as st, sys
from collections import Counter

THINK = re.compile(r"<think>(.*?)</think>", re.S)

ASSISTANT_VOICE = re.compile(
    r"\b(the user (wants|is asking|asked|has asked)|the (task|prompt|question) (asks|is|says)"
    r"|i need to (fill|write|implement|produce|create) (in |out )?(the|a) (scaffold|stub|code|function)"
    r"|as an ai|my task is to|i (should|must) (follow|respect) the (format|contract|instruction))\b", re.I)
SCIENTIST_VOICE = re.compile(
    r"\b(let me think|from scratch|the thing everyone|i want to be (very )?precise"
    r"|so picture|here'?s what bothers me|my worry is|the reason i|i keep coming back to"
    r"|what i actually (want|need) to know|suppose|imagine)\b", re.I)
HEDGE = re.compile(
    r"\b(i(?:'m| am) not (sure|certain)|not confident|uncertain|might not|may not|"
    r"could be wrong|hard to say|unclear whether|i don'?t (know|actually know)|"
    r"remains? (unclear|open|untested)|no guarantee)\b", re.I)
DEADEND = re.compile(
    r"\b(does ?n'?t work|did ?n'?t work|dead end|abandon(ed)?|gave? up on|"
    r"that fails|ruled out|discard(ed)?|scrap(ped)? that|backtrack)\b", re.I)
STOP = set("""the a an and or of to in is are was were be been it its this that these those for on with as at by
from we our i my you your they their he she but if then so than not no do does did have has had will would can
could should may might must there here what which who whom whose when where why how all any both each few more
most other some such only own same s t just don now""".split())


def words(s):
    return [w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOP]


def answer_key_terms(answer, prompt):
    """The answer's own recurring content words that the prompt did NOT supply."""
    a = Counter(words(answer))
    key = {w for w, c in a.items() if c >= 2}
    return key - set(words(prompt))


def leak_profile(think, answer, prompt):
    """(front_loading, total) coverage of the answer's prompt-absent key terms.

    front_loading = share already present in the think's first fifth. High total is
    expected and fine (the reasoning reaches the answer); high front-loading is the
    signature of restating a known answer instead of deriving it.
    """
    key = answer_key_terms(answer, prompt)
    if len(key) < 8 or len(think) < 500:
        return None
    q1 = set(words(think[: len(think) // 5]))
    allw = set(words(think))
    return len(q1 & key) / len(key), len(allw & key) / len(key)


def _prompt_of(r):
    return " ".join([r.get("system") or ""] +
                    [m["value"] for m in r["conversations"]
                     if m["from"] in ("human", "observation")])


def turns(path):
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        rid = r.get("_id")
        pr = _prompt_of(r)
        for j, m in enumerate(r["conversations"]):
            if m["from"] not in ("gpt", "function_call") or not m.get("loss"):
                continue
            mt = THINK.search(m["value"])
            if not mt:
                continue
            yield (rid, j), mt.group(1), m["value"][mt.end():], pr


def rates(items):
    n = len(items)
    if n == 0:
        return {}
    L = [len(t) for _, t, _, _ in items]
    prof = [p for p in (leak_profile(t, a, pr) for _, t, a, pr in items) if p is not None]
    f = lambda rx: sum(1 for _, t, _, _ in items if rx.search(t)) / n * 100
    out = {
        "n": n,
        "chars_median": st.median(L),
        "assistant_voice_%": f(ASSISTANT_VOICE),
        "scientist_voice_%": f(SCIENTIST_VOICE),
        "hedges_%": f(HEDGE),
        "mentions_dead_end_%": f(DEADEND),
    }
    if prof:
        out["leak_front_loading_median"] = st.median([p[0] for p in prof])
        out["leak_frontloaded_over_0.5_%"] = sum(p[0] > 0.5 for p in prof) / len(prof) * 100
        out["answer_terms_reached_median"] = st.median([p[1] for p in prof])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl")
    ap.add_argument("--distill", default=".cache/tinker/innovation_distilled.jsonl")
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()

    dist = {k: (t, ans, pr) for k, t, ans, pr in turns(a.distill)}
    # compare only on the turns that were actually regenerated
    keep = set(dist)
    # the distilled file carries _id = source line index, so re-walk the source with it
    orig_by_key = {}
    for i, line in enumerate(open(a.orig)):
        rid = f"{i:05d}"
        r = json.loads(line)
        for j, m in enumerate(r["conversations"]):
            if m["from"] not in ("gpt", "function_call") or not m.get("loss"):
                continue
            mt = THINK.search(m["value"])
            if mt and (rid, j) in keep:
                orig_by_key[(rid, j)] = (mt.group(1), m["value"][mt.end():], _prompt_of(r))

    common = sorted(set(dist) & set(orig_by_key))
    print(f"[qc] {len(common)} regenerated turns matched to their originals\n")
    O = [(k, *orig_by_key[k]) for k in common]
    D = [(k, *dist[k]) for k in common]
    ro, rd = rates(O), rates(D)
    print(f"{'metric':30s} {'hand-written':>14} {'distilled':>12}")
    for k in ro:
        vo, vd = ro[k], rd[k]
        fmt = (lambda v: f"{v:.3f}") if isinstance(vo, float) and vo < 5 else (lambda v: f"{v:,.1f}")
        print(f"{k:30s} {fmt(vo):>14} {fmt(vd):>12}")

    print("\n--- side-by-side openings ---")
    for k in common[: a.samples]:
        print(f"\n### turn {k}")
        print("  ORIG : " + orig_by_key[k][0].strip()[:300].replace("\n", " "))
        print("  DIST : " + dist[k][0].strip()[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
