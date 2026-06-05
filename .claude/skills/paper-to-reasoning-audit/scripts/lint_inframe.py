#!/usr/bin/env python3
"""Deterministic in-frame lint for paper-to-reasoning deliverables.

Catches the enumerable rule-violation classes the skill's 2.4 pass is supposed to
strip but a judgment-based audit misses. Runs over every <root>/<slug>/results/*.md.

Usage:
  lint_inframe.py            # report mode (scans methods/ + top-level new dirs)
  lint_inframe.py --fix      # also auto-fix the mechanical classes (A, and reasoning headers)
"""
import os, re, sys, glob

FIX = "--fix" in sys.argv

# ---- pattern classes -------------------------------------------------------
# A) parenthetical self-compliance annotations (banned anywhere) -- AUTO-FIXABLE
PAREN_TIC = re.compile(
    r"\s*\((?:faithful[^)]*|known(?:\s+[a-z]+)?|pre-method|"
    r"settings only[^)]*|no outcomes[^)]*|no results[^)]*|"
    r"known recipes?[^)]*|known primitives?[^)]*)\)", re.IGNORECASE)

# B) meta / self-compliance prose lines (review) -- NOT auto-fixed
META_PROSE = re.compile(
    r"No outcomes are stated|settings only,?\s*(?:no|—)|strictly pre-method|"
    r"Known primitives only|no naming of the method|no results are reported|"
    r"No official repositor|no official implementation|reference implementation|"
    r"official repo\b|official repository", re.IGNORECASE)

# E) target-paper self-reference (review)
PAPER_REF = re.compile(r"\bthis paper\b|\bthe paper\b|\bthe authors\b", re.IGNORECASE)

# D) CJK / stray non-ASCII letters (review)
CJK = re.compile(r"[一-鿿぀-ヿ]")

HEADER = re.compile(r"^\s{0,3}#{1,6}\s")
FENCE = re.compile(r"^\s*```")

def scan(path):
    """Return dict category -> list[(lineno, text)] and the (possibly fixed) lines."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    hits = {"A_paren": [], "B_meta": [], "E_paperref": [], "D_cjk": [], "C_rsn_header": []}
    is_reasoning = path.endswith("reasoning.md")
    in_fence = False
    out = []
    for i, ln in enumerate(lines, 1):
        if FENCE.match(ln):
            in_fence = not in_fence
        # A) parenthetical tic
        if PAREN_TIC.search(ln):
            hits["A_paren"].append((i, ln.rstrip()))
            if FIX:
                ln = PAREN_TIC.sub("", ln)
        # B) meta prose
        if META_PROSE.search(ln):
            hits["B_meta"].append((i, ln.rstrip()))
        # E) paper ref
        if PAPER_REF.search(ln):
            hits["E_paperref"].append((i, ln.rstrip()))
        # D) cjk
        if CJK.search(ln):
            hits["D_cjk"].append((i, ln.rstrip()))
        # C) markdown header inside reasoning.md PROSE (outside code fence)
        if is_reasoning and not in_fence and HEADER.match(ln):
            hits["C_rsn_header"].append((i, ln.rstrip()))
        out.append(ln)
    return hits, out, lines

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots = sorted(set(
        glob.glob(os.path.join(base, "methods", "*", "results")) +
        [d for d in glob.glob(os.path.join(base, "*", "results"))
         if os.path.basename(os.path.dirname(d)) not in ("methods",)]))
    totals = {k: 0 for k in ["A_paren", "B_meta", "E_paperref", "D_cjk", "C_rsn_header"]}
    files_with_hits = 0
    for results_dir in roots:
        for md in sorted(glob.glob(os.path.join(results_dir, "*.md"))):
            hits, out, orig = scan(md)
            n = sum(len(v) for v in hits.values())
            if n == 0:
                continue
            files_with_hits += 1
            rel = os.path.relpath(md, base)
            for cat, items in hits.items():
                for lineno, text in items:
                    totals[cat] += 1
                    print(f"{cat:14} {rel}:{lineno}: {text.strip()[:140]}")
            if FIX and out != orig:
                with open(md, "w", encoding="utf-8") as f:
                    f.writelines(out)
    print("\n=== totals ===")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    print(f"  files with hits: {files_with_hits}")

if __name__ == "__main__":
    main()
