#!/usr/bin/env python3
"""Audit within-family reuse: are same-family problems the SAME problem re-skinned?

Statement wording is a weak signal — two problems can read differently and still
optimize the identical objective under identical constraints. The objective and the
constraint set live in the checker (chk.cc / verify.py / counter.py / evaluator.py),
so that is what we compare.

Method (standard code-clone detection): strip comments, string literals and numeric
literals, tokenize, take token 5-grams, Jaccard. Numeric literals are stripped so that
"same logic, retuned constants" still registers as a clone — that is exactly the
re-skinning failure mode we are hunting.

Reported per family:
  chk_med   median pairwise checker 5-gram Jaccard  <- the decisive number
  stm_med   median pairwise statement token Jaccard <- wording only
  n_hi      pairs with chk >= CLONE_HI (near-identical scoring logic)
"""
import json, os, re, sys, itertools, statistics
from collections import defaultdict

SYNTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONE_HI = 0.60      # >= this on checker 5-grams == effectively the same scorer
CLONE_MID = 0.40

CHECKERS = ("chk.cc", "verify.py", "counter.py", "evaluator.py")
STATEMENTS = ("statement.txt", "statement.md")

def _read(pdir, names):
    for n in names:
        p = os.path.join(pdir, n)
        if os.path.isfile(p):
            try:
                return open(p, encoding="utf-8", errors="replace").read(), n
            except Exception:
                pass
    return "", None

def norm_code(src):
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    src = re.sub(r'//[^\n]*', ' ', src)
    src = re.sub(r'#[^\n]*', ' ', src)              # py comments + cpp preprocessor
    src = re.sub(r'"""[\s\S]*?"""', ' ', src)
    src = re.sub(r"'''[\s\S]*?'''", ' ', src)
    src = re.sub(r'"(?:\\.|[^"\\])*"', ' STR ', src)
    src = re.sub(r"'(?:\\.|[^'\\])*'", ' STR ', src)
    src = re.sub(r'\b\d+\.?\d*(?:[eE][-+]?\d+)?\b', ' NUM ', src)
    return re.findall(r'[A-Za-z_]\w*|[^\sA-Za-z_\w]', src)

def grams(tokens, n=5):
    return {tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))}

def jac(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def main():
    rows = [json.loads(l) for l in open(os.path.join(SYNTH, "seeds/seed_list.jsonl"))]
    fam = defaultdict(list)
    for d in rows:
        fam[d["family"]].append(d["id"])

    big = sorted([(f, ids) for f, ids in fam.items() if len(ids) >= 2],
                 key=lambda kv: -len(kv[1]))
    cache_c, cache_s = {}, {}

    def cg(pid):
        if pid not in cache_c:
            src, _ = _read(os.path.join(SYNTH, "problems", pid), CHECKERS)
            cache_c[pid] = grams(norm_code(src))
        return cache_c[pid]

    def sg(pid):
        if pid not in cache_s:
            txt, _ = _read(os.path.join(SYNTH, "problems", pid), STATEMENTS)
            cache_s[pid] = set(re.findall(r'[a-z]{4,}', txt.lower()))
        return cache_s[pid]

    print(f"{'family':38s} {'n':>3s} {'chk_med':>8s} {'chk_max':>8s} {'stm_med':>8s} {'n_hi':>5s}  verdict")
    print("-" * 92)
    flagged = []
    for f, ids in big:
        cs, ss, hi, worst = [], [], 0, (0.0, None, None)
        for a, b in itertools.combinations(sorted(ids), 2):
            c = jac(cg(a), cg(b))
            cs.append(c); ss.append(jac(sg(a), sg(b)))
            if c >= CLONE_HI: hi += 1
            if c > worst[0]: worst = (c, a, b)
        if not cs: continue
        cm, sm = statistics.median(cs), statistics.median(ss)
        npairs = len(cs)
        if cm >= CLONE_HI:      v = "CLONE-FAMILY"
        elif cm >= CLONE_MID:   v = "heavy reuse"
        elif hi:                v = f"{hi} clone pair(s)"
        else:                   v = "ok"
        if v != "ok":
            flagged.append((f, len(ids), cm, worst, v))
        print(f"{f[:38]:38s} {len(ids):3d} {cm:8.3f} {worst[0]:8.3f} {sm:8.3f} {hi:5d}  {v}")

    print()
    print(f"families with >=2 problems: {len(big)}")
    print(f"flagged: {len(flagged)}")
    if flagged:
        print("\nworst offending pair per flagged family:")
        for f, n, cm, (w, a, b), _v in sorted(flagged, key=lambda x: -x[3][0])[:15]:
            print(f"  {f[:34]:34s} n={n:3d} chk_med={cm:.3f}  worst {a} <-> {b} = {w:.3f}")

if __name__ == "__main__":
    main()
