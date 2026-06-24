#!/usr/bin/env python3
"""Append a 'commit-discipline' coda to code-bearing landing points (DATA_REMEDIATION_zh.md §3-A2).

The FCS/ALE regression traces to a missing skill: the data never demonstrates "converge -> (if the
fancy idea isn't safe) fall back to the simplest correct version -> ship a single self-contained,
self-checked deliverable". This tool injects that decision structure as a short, trained coda placed
AFTER the method's final code, so every code-bearing sample carries the landing register.

It is a CHEAP language/decision anchor, NOT the main fix -- high-quality fallback+land behavior should
come from the new trajectories in track B. The coda is grounded in the sample's own final code block
(it names the language and restates the ship/self-check/fallback decision), so it is not pure boiler-
plate, but it is deliberately generic. Apply it preferentially to competition/optimization methods.

SAFE BY DEFAULT: dry-run, writes previews to sft/coda/<slug>-coda.md and never touches the originals.
Pass --apply to append the coda to methods/<slug>/results/train_answer.md in place (idempotent: it
skips files that already contain the coda marker).

Usage:
  python tools/make_commit_coda.py                 # preview, competition/optimization categories
  python tools/make_commit_coda.py --all-categories # preview for every code-bearing method
  python tools/make_commit_coda.py --apply          # write in place (idempotent)
"""
import glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER = "<!-- commit-coda v1 -->"
TARGET_CATEGORIES = {"Combinatorial & Competitive Algorithms"}
TARGET_DOMAINS = {"Optimization", "Informatics olympiad", "Combinatorial optimization",
                  "Geometric optimization", "Heuristic combinatorial optimization", "Fast algorithms"}

LANG_FENCE = re.compile(r'```([A-Za-z+]+)?\s*\n(.*?)```', re.DOTALL)


def last_code_lang(text):
    blocks = LANG_FENCE.findall(text or '')
    if not blocks:
        return None
    lang = (blocks[-1][0] or '').strip().lower() or 'code'
    return {'cpp': 'C++', 'c++': 'C++', 'cc': 'C++', 'python': 'Python', 'py': 'Python'}.get(lang, lang)


def coda_for(lang):
    """A short, first-person commit-discipline coda referencing the deliverable's language."""
    return (
        f"\n\n{MARKER}\n"
        f"That is the method. Before I call it done I want this to be an actual deliverable, not a "
        f"sketch: one self-contained {lang} program that respects the input/output contract exactly "
        f"and runs end to end. So I read the final code back as a submission — does it parse its input, "
        f"handle the empty / single-element / maximum-size cases, and print exactly the required output "
        f"and nothing else? If any part of the clever construction above is something I am not certain "
        f"I can get correct within the budget, I do not gamble on it: I fall back to the simplest version "
        f"I have already convinced myself is correct and ship that instead — a plain correct solution "
        f"beats an ambitious broken one, which on a scored task can do worse than doing nothing. I run "
        f"it once in my head on the smallest example, confirm the output matches, and submit that."
    )


def main():
    apply = '--apply' in sys.argv
    all_cats = '--all-categories' in sys.argv
    methods = {m['slug']: m for m in json.load(open(os.path.join(ROOT, 'methods.json')))}
    os.makedirs(os.path.join(ROOT, 'sft/coda'), exist_ok=True)

    considered = wrote = skipped_nocode = skipped_marker = skipped_cat = 0
    for p in sorted(glob.glob(os.path.join(ROOT, 'methods/*/results/train_answer.md'))):
        slug = p.split('/methods/')[1].split('/')[0]
        m = methods.get(slug, {})
        in_scope = all_cats or m.get('category') in TARGET_CATEGORIES or m.get('domain') in TARGET_DOMAINS
        if not in_scope:
            skipped_cat += 1
            continue
        considered += 1
        text = open(p, encoding='utf-8').read()
        if MARKER in text:
            skipped_marker += 1
            continue
        lang = last_code_lang(text)
        if lang is None:
            skipped_nocode += 1
            continue
        coda = coda_for(lang)
        if apply:
            open(p, 'a', encoding='utf-8').write(coda)
        else:
            open(os.path.join(ROOT, 'sft/coda', f'{slug}-coda.md'), 'w', encoding='utf-8').write(coda.strip())
        wrote += 1

    mode = 'APPLIED to train_answer.md' if apply else 'previewed to sft/coda/'
    print(f"scope: {'all code-bearing methods' if all_cats else 'competition/optimization categories'}")
    print(f"  considered: {considered} | coda {mode}: {wrote} | "
          f"skipped(no code): {skipped_nocode} | skipped(already has coda): {skipped_marker} | "
          f"out-of-scope: {skipped_cat}")
    if not apply:
        print("  (dry run — re-run with --apply to write in place; idempotent via the coda marker)")


if __name__ == '__main__':
    main()
