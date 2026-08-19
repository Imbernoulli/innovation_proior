#!/usr/bin/env python3
"""Scan single-turn method units for self-supplied observations (proposal-frame violations).

V1: narrator claims to run/train/measure and states the outcome.
V2: own-method results reported as accomplished ("our experiments show", "achieves 78.9 top-1").
Heuristic co-occurrence at paragraph level; output JSONL of hits with quotes for review.
"""
import json, os, re, sys, glob
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- pattern pieces -----------------------------------------------------------
RUN_VERBS = r"(?:run|ran|rerun|train(?:ed)?|retrain(?:ed)?|fine-?tun(?:e|ed)|test(?:ed)?|tr(?:y|ied)|sweep|swept|benchmark(?:ed)?|evaluat(?:e|ed)|measur(?:e|ed)|profil(?:e|ed)|ablat(?:e|ed)|launch(?:ed)?|fit(?:ted)?)"
FIRST = r"(?:I|we)\s+(?:then\s+|also\s+|now\s+|actually\s+)?" + RUN_VERBS
RESULT_WORDS = r"(?:comes?\s+back|returns?|reports?|yields?|gives?|shows?|observ(?:e|ed)|recorded|measured|lands?\s+at|drops?\s+to|rises?\s+to|reaches|converg|diverg|scores?|top-?1|top-?5|accuracy|BLEU|FID|WER|perplexity|val(?:idation)?\s+(?:loss|error)|test\s+(?:error|accuracy)|win\s+rate|success\s+rate)"
NUM = r"\d+(?:\.\d+)?\s*%?"
EXPT_NOUN = r"(?:ablation|experiment|sweep|run|trial|benchmark|pilot|prototype|training\s+run)"

PATTERNS = [
    # V1a: "I run/train ... <result-word or number>" in same paragraph handled by co-occurrence below
    ("V1_first_person_run", re.compile(FIRST, re.I)),
    ("V1_expt_shows", re.compile(EXPT_NOUN + r"s?\s+(?:shows?|gives?|returns?|comes?\s+back|confirms?|reveals?|says)", re.I)),
    ("V1_when_i_run", re.compile(r"(?:when|after|once)\s+(?:I|we)\s+" + RUN_VERBS, re.I)),
    ("V1_observed", re.compile(r"(?:I|we)\s+(?:observe|observed|find|found|see|saw|got|get|obtain(?:ed)?)\s+(?:that\s+)?(?:[^.\n]{0,80}?)" + NUM, re.I)),
    ("V2_our_expts", re.compile(r"(?:our|my)\s+(?:experiments?|runs?|ablations?|results?)\s+(?:show|demonstrate|confirm|indicate)", re.I)),
    ("V2_achieves", re.compile(r"(?:achiev(?:e|es|ed)|reach(?:es|ed)?|obtains?|gets?\s+to|attains?)\s+" + NUM + r"\s*(?:top-?1|top-?5|accuracy|BLEU|FID|WER|F1|AUC|mAP|dB|points?)", re.I)),
]
RESULT_RE = re.compile(RESULT_WORDS, re.I)
NUM_RE = re.compile(NUM)

# benign guards: prediction/plan language and prior-work attribution nearby
GUARD = re.compile(r"\b(?:would|should|expect|predict|plan|will\s+decide|is\s+the\s+test|to\s+decide|hypothes|propose|design|reported\s+(?:in|by)|published|known\s+(?:result|value)|literature|prior\s+work|earlier\s+work|they\s+report(?:ed)?)\b", re.I)

def paragraphs(text):
    buf, in_code = [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.strip() == "":
            if buf:
                yield " ".join(buf); buf = []
        else:
            buf.append(line.strip())
    if buf:
        yield " ".join(buf)

def scan_file(path):
    hits = []
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return hits
    for para in paragraphs(text):
        matched = []
        for name, rx in PATTERNS:
            m = rx.search(para)
            if not m:
                continue
            # V1_first_person_run alone is weak: require result-language or number in the paragraph
            if name == "V1_first_person_run" and not (RESULT_RE.search(para) or NUM_RE.search(para)):
                continue
            matched.append((name, m))
        if not matched:
            continue
        guard = bool(GUARD.search(para))
        for name, m in matched:
            a = max(0, m.start() - 60)
            b = min(len(para), m.end() + 140)
            hits.append({"pattern": name, "guarded": guard,
                         "quote": para[a:b].strip()})
    return hits

def main():
    out_path = os.path.join(ROOT, "experiments", "source_value_audit", "obs_scan_hits.jsonl")
    stats = Counter(); unit_stats = Counter()
    n_units = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for d in sorted(glob.glob(os.path.join(ROOT, "methods", "*", "results"))):
            slug = os.path.basename(os.path.dirname(d))
            n_units += 1
            unit_hits = []
            for fname in ("reasoning.md", "answer.md", "train_answer.md"):
                for h in scan_file(os.path.join(d, fname)):
                    h.update(slug=slug, file=fname.split(".")[0])
                    unit_hits.append(h)
            for h in unit_hits:
                stats[(h["pattern"], h["guarded"])] += 1
                out.write(json.dumps(h, ensure_ascii=False) + "\n")
            if unit_hits:
                unguarded = [h for h in unit_hits if not h["guarded"]]
                unit_stats["any"] += 1
                if unguarded:
                    unit_stats["unguarded"] += 1
    print(f"units scanned: {n_units}; units with hits: {unit_stats['any']}; with unguarded hits: {unit_stats['unguarded']}")
    for (p, g), c in sorted(stats.items()):
        print(f"  {p:24s} guarded={g}  {c}")
    print("hits ->", out_path)

if __name__ == "__main__":
    main()
