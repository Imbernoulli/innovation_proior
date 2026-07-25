#!/usr/bin/env python3
"""Report whether a method's train_answer.md passes the build-time code-integrity gate.

This is the SAME predicate sft/build_sft.py uses to decide the answer channel: train_answer.md
is trained only if every large code fence in it is a whitespace-normalized substring of
answer.md's fences (i.e. copied verbatim, per the discovery-writeup contract); otherwise the
build falls back to presenting answer.md itself.

Usage:
  python3 tools/ta_gate_check.py <slug> [<slug> ...]      # methods/<slug>/results
  python3 tools/ta_gate_check.py --all                     # every method in methods.json
  python3 tools/ta_gate_check.py --path <dir>              # any dir holding answer/train_answer
Exit code 0 iff every checked unit PASSes.
"""
import json, os, re, sys

_FENCE_RE = re.compile(r'```[a-zA-Z0-9_+-]*\n(.*?)```', re.S)


def fences(text):
    if text.count('```') % 2:          # unclosed final fence: markdown closes it at EOF
        text = text + '\n```'
    return _FENCE_RE.findall(text)


def gate(ta_text, ans_text):
    """(ok, reason) — mirrors build_sft._ta_code_ok."""
    for bad in ('~~~', '<pre>', '</code>'):
        if bad in ta_text:
            return False, f'non-backtick code carrier {bad!r} (fail-closed)'
    hay = re.sub(r'\s+', '', ''.join(fences(ans_text)))
    small = 0
    for i, b in enumerate(fences(ta_text)):
        nb = re.sub(r'\s+', '', b)
        if nb in hay:
            continue
        if len(nb) >= 200:
            head = b.strip().splitlines()[:1]
            return False, f'fence #{i + 1} ({len(nb)} chars) is not verbatim from answer.md: {head}'
        small += len(nb)
    if small >= 200:
        return False, f'divergent small fences aggregate to {small} chars'
    return True, 'every code fence is verbatim from answer.md (or there is no code)'


def check_dir(d, label=None):
    label = label or d
    ta_p, ans_p = f'{d}/train_answer.md', f'{d}/answer.md'
    if not os.path.isfile(ta_p):
        return False, f'{label}: MISSING train_answer.md (build falls back to answer.md)'
    if not os.path.isfile(ans_p):
        return True, f'{label}: PASS (no answer.md to compare against)'
    ok, why = gate(open(ta_p, encoding='utf-8').read(), open(ans_p, encoding='utf-8').read())
    return ok, f'{label}: {"PASS" if ok else "FAIL"} — {why}'


def main(argv):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == '--all':
        targets = [(f"methods/{m['slug']}/results", m['slug']) for m in json.load(open('methods.json'))]
        targets = [t for t in targets if os.path.isdir(t[0])]
    elif argv[0] == '--path':
        targets = [(p, p) for p in argv[1:]]
    else:
        targets = [(f'methods/{s}/results', s) for s in argv]
    bad = 0
    for d, label in targets:
        ok, msg = check_dir(d, label)
        if not ok:
            bad += 1
        if not ok or len(targets) <= 20:
            print(msg)
    if len(targets) > 20:
        print(f'{len(targets) - bad}/{len(targets)} PASS, {bad} FAIL')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
