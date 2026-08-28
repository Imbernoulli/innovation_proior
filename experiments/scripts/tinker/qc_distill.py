"""QC the distilled corpus against the hand-written original, turn by turn.

Three things we actually care about, in priority order:

1. VOICE. Our corpus's one distinctive asset is the first-person scientist
   ("OK, let me think this through from scratch..."). If the teacher's own
   assistant register ("The user wants me to...") replaces it, the distillation
   fixed the token distribution and threw away the thing worth keeping.

2. HINDSIGHT LEAK. The teacher was trained on (prompt -> think + answer), so when
   it regenerates the think from the prompt alone it may simply restate the answer
   it memorised. That would make the hindsight problem WORSE, not better. Measured
   as content-word overlap between the regenerated think and the answer that was
   NOT in its prompt.

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


def overlap(think, answer):
    """Fraction of the answer's distinctive content words that the think already uses."""
    a = Counter(words(answer)); t = set(words(think))
    a = {w: c for w, c in a.items() if c >= 2}          # answer's own recurring terms
    if not a:
        return None
    return sum(1 for w in a if w in t) / len(a)


def turns(path, key_id=None):
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        rid = r.get("_id")
        for j, m in enumerate(r["conversations"]):
            if m["from"] not in ("gpt", "function_call") or not m.get("loss"):
                continue
            mt = THINK.search(m["value"])
            if not mt:
                continue
            yield (rid, j), mt.group(1), m["value"][mt.end():]


def rates(items):
    n = len(items)
    if n == 0:
        return {}
    L = [len(t) for _, t, _ in items]
    ov = [o for o in (overlap(t, a) for _, t, a in items) if o is not None]
    f = lambda rx: sum(1 for _, t, _ in items if rx.search(t)) / n * 100
    return {
        "n": n,
        "chars_median": st.median(L),
        "assistant_voice_%": f(ASSISTANT_VOICE),
        "scientist_voice_%": f(SCIENTIST_VOICE),
        "hedges_%": f(HEDGE),
        "mentions_dead_end_%": f(DEADEND),
        "answer_term_overlap_median": st.median(ov) if ov else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl")
    ap.add_argument("--distill", default=".cache/tinker/innovation_distilled.jsonl")
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()

    dist = {k: (t, ans) for k, t, ans in turns(a.distill)}
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
                orig_by_key[(rid, j)] = (mt.group(1), m["value"][mt.end():])

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
