#!/usr/bin/env python3
"""
Prompt-side CONCEPTUAL-LEAKAGE flagger for context.md (the SFT prompt).

Why this exists: the in-frame lint (lint_inframe.py) catches *lexical* leaks — the
method's NAME, "this paper", meta tags. But a context.md can leak the method's
central MOVE without ever naming it: a baseline gap written as a prescription of
the fix ("one needs the analogue over Z_q"), or a scaffold TODO that pre-names the
target component ("# TODO: provable(x)=∃y proof_of") or pre-locates the fix
("if gradient_is_small: <slot>"). A 6-method spot audit found this leak in 6/6 —
systematic, and invisible to both the regex lint and the Codex math/in-frame gate.

This is a SUSPECT FLAGGER, not a verdict. Conceptual leakage is semantic; regex
can only surface high-risk phrasings. Treat every hit as "send to the LLM-judge"
(audit SKILL.md Layer-2 'prompt-side leakage' rubric), not as a confirmed defect.
Conversely, zero hits does NOT mean clean — a leak can be paraphrased past these
patterns; the judge is the authority.

Usage:
  python detect_leakage.py [BASE]            # report (BASE defaults to cwd)
  python detect_leakage.py [BASE] --slugs a,b,c   # only these methods
Scans <BASE>/methods/<slug>/results/context.md.
"""
import os, re, sys, glob

# --- Channel A: a baseline gap stated as a PRESCRIPTION of the fix (not an
#     observed limitation). These phrasings hand the method's own move into the prompt.
PRESCRIPTIVE = re.compile(
    r"\bone needs?\b"
    r"|\bwhat(?:'s| is) (?:missing|needed|required|lacking)\b"
    r"|\bwhat (?:we|one) (?:need|want|require)s? (?:is|here)\b"
    r"|\bthe analogue (?:over|of|for)\b"
    r"|\bthe quantity that (?:would|might|could|will|may) replace\b"
    r"|\bthe (?:missing|needed|required|right) (?:ingredient|piece|object|quantity|tool|move|notion|quantity|measure)\b"
    r"|\b(?:this|it|that) (?:calls for|cries out for|demands|requires) (?:a|an|the|introducing|defining|building)\b"
    r"|\bmust (?:introduce|invent|add|define|construct|build|replace) (?:a|an|the)\b"
    r"|\bthe (?:key|trick|fix|cure|remedy|solution) (?:is|here is|would be) (?:to|going|a|an|the)\b"
    r"|\bwould (?:replace|substitute for|stand in for) (?:N|the (?:cardinality|union bound|enumeration))\b",
    re.I,
)

# --- Channel B: a scaffold TODO that is NOT neutral — it pre-names the target
#     component (math symbols / a specific object) or pre-locates the contribution.
TODO_LINE = re.compile(r"#\s*TODO\b", re.I)
# A neutral TODO is short and generic ("the architecture we'll design",
# "the object we'll define here"). Flag a TODO if it carries math/answer content:
TODO_LEAKY = re.compile(
    r"[=∃∀∑∫]|:=|\\frac|\bif\b.*\b(small|stuck|near|stationary|saddle|zero)\b"
    r"|\bprovable\b|\bproof_of\b|\bvalue function\b|\bper-slice\b|\beligibility\b"
    r"|\bposterior\b|\bbonus\b|\bperturb",
    re.I,
)
GENERIC_TODO_OK = re.compile(
    r"the (?:architecture|object|method|thing|piece|construction|update|rule|model|quantity) "
    r"(?:we'll|we will|to) (?:design|define|build|discover|construct)",
    re.I,
)


def scan_context(path):
    hits = {"A_prescriptive_gap": [], "B_leaky_scaffold_todo": []}
    in_fence = False
    for i, ln in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if PRESCRIPTIVE.search(ln):
            hits["A_prescriptive_gap"].append((i, ln.rstrip()))
        if TODO_LINE.search(ln) and TODO_LEAKY.search(ln) and not GENERIC_TODO_OK.search(ln):
            hits["B_leaky_scaffold_todo"].append((i, ln.rstrip()))
    return hits


def main():
    args = [a for a in sys.argv[1:]]
    base = "."
    slugs = None
    if args and not args[0].startswith("--"):
        base = args[0]; args = args[1:]
    if "--slugs" in args:
        slugs = set(args[args.index("--slugs") + 1].split(","))
    paths = []
    for cm in sorted(glob.glob(os.path.join(base, "methods", "*", "results", "context.md"))):
        slug = os.path.basename(os.path.dirname(os.path.dirname(cm)))
        if slugs is None or slug in slugs:
            paths.append((slug, cm))
    totals = {"A_prescriptive_gap": 0, "B_leaky_scaffold_todo": 0}
    flagged = []
    for slug, cm in paths:
        h = scan_context(cm)
        n = sum(len(v) for v in h.values())
        for k in totals:
            totals[k] += len(h[k])
        if n:
            flagged.append((slug, h))
    print(f"scanned context.md: {len(paths)} | flagged (regex suspects): {len(flagged)}")
    print(f"totals: {totals}")
    for slug, h in flagged:
        print(f"\n# {slug}")
        for cat, items in h.items():
            for i, ln in items:
                print(f"  [{cat}] L{i}: {ln.strip()[:140]}")
    print("\nNOTE: regex SUSPECTS only — confirm/deny with the LLM-judge (audit "
          "SKILL.md → 'prompt-side conceptual leakage' rubric). Zero hits != clean.")


if __name__ == "__main__":
    main()
