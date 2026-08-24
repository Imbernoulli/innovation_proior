#!/usr/bin/env python3
"""Apply a prose-fills file onto an agentic_v2.json skeleton, then lint it.

The prose workflow NEVER edits agentic_v2.json directly (its edit ops are
byte-verified replay state — one stray change corrupts the episode). Instead the
filling agent writes trajectories/<task>/agentic_v2_fills.json:

  {"task": "<task>",
   "fills": {"THINK kind=design rung=1 slug=gin seq=1": "<the think text>",
             "SAY rung=1 slug=gin seq=1": "<one-two sentences>",
             ...}}

and this script substitutes every `[[<key>]]` sentinel with its fill, enforcing:
  * every sentinel has a fill and every fill matches a sentinel (no drift);
  * every think is non-empty prose (>= 40 chars; the whole point of v2 is that NO
    trained turn carries an empty think), <= 12000 chars hard cap;
  * fills carry no sentinel syntax, no chat-template literals (<think>, <tool_call>
    — build_sft neutralizes, but the fill should never rely on that), and no
    repo-internal paths (results/..., methods/<slug>/ — the trained channel must
    stay in-frame);
  * pre_test fills do not quote the still-unseen result of the test they precede
    (checked crudely: no digit-bearing metric value that appears ONLY in that
    test's feedback and nowhere earlier — full judgment stays with the reviewer);
  * after substitution, tools/build_agentic_v2.py --lint passes (replay + no
    sentinel + no empty think).

Usage:
  python3 tools/apply_agentic_fills.py TASK [TASK...]   # apply + lint
  python3 tools/apply_agentic_fills.py --check TASK     # validate only, no write
"""
import json
import os
import re
import subprocess
import sys

REPO = os.environ.get('INNOVATION_PRIOR_REPO') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

SENT = re.compile(r'\[\[([^\]]+)\]\]')
BAD_LITERALS = ('<think>', '</think>', '<tool_call>', '</tool_call>',
                '<tool_response>', '<|im_start|>', '<|im_end|>', '[[', ']]')
LEAK = re.compile(r'\b(results/[a-z0-9_./-]+|methods/[a-z0-9-]+/|trajectories/[a-z0-9-]+/'
                  r'|reasoning\.md|context\.md|train_answer\.md|answer\.md)')


def fail(msgs):
    for m in msgs:
        print('FILLS-ERROR:', m)
    return False


def apply_task(task, check_only=False):
    d = f'trajectories/{task}'
    skel = json.load(open(f'{d}/agentic_v2.json', encoding='utf-8'))
    fp = f'{d}/agentic_v2_fills.json'
    if not os.path.isfile(fp):
        return fail([f'{task}: missing {fp}'])
    fills = json.load(open(fp, encoding='utf-8'))['fills']

    slots = []
    for m in skel['messages']:
        if m['role'] != 'assistant':
            continue
        for field in ('reasoning_content', 'content'):
            v = m.get(field) or ''
            ms = SENT.fullmatch(v.strip())
            if ms:
                slots.append(ms.group(1))
    errs = []
    missing = [s for s in slots if s not in fills]
    extra = [k for k in fills if k not in slots]
    if missing:
        errs.append(f'{task}: {len(missing)} slot(s) missing a fill, e.g. {missing[:3]}')
    if extra:
        errs.append(f'{task}: {len(extra)} fill(s) match no slot, e.g. {extra[:3]}')
    for k, v in fills.items():
        if not isinstance(v, str):
            errs.append(f'{k}: fill is not a string'); continue
        t = v.strip()
        if k.startswith('THINK') and len(t) < 40:
            errs.append(f'{k}: think fill too short ({len(t)} chars) — every trained turn must carry real reasoning')
        if k.startswith('SAY') and not t:
            errs.append(f'{k}: SAY fill empty')
        if len(t) > 12000:
            errs.append(f'{k}: fill exceeds 12000-char hard cap ({len(t)})')
        for lit in BAD_LITERALS:
            if lit in t:
                errs.append(f'{k}: fill contains forbidden literal {lit!r}')
                break
        lk = LEAK.search(t)
        if lk:
            errs.append(f'{k}: repo-internal path leak {lk.group(0)!r}')
    if errs:
        return fail(errs)

    # crude future-leak check for pre_test fills: a metric-looking number that first
    # appears in the upcoming test result must not already be in the think.
    NUM = re.compile(r'\d+\.\d{2,}')
    msgs = skel['messages']
    seen = []
    for i, m in enumerate(msgs):
        if m['role'] == 'assistant':
            v = (m.get('reasoning_content') or '').strip()
            ms = SENT.fullmatch(v)
            text = fills[ms.group(1)] if ms else v
            if ms and ms.group(1).startswith('THINK kind=pre_test'):
                nxt = msgs[i + 1]['content'] if i + 1 < len(msgs) else ''
                fresh = set(NUM.findall(nxt)) - set(NUM.findall('\n'.join(seen)))
                hit = [x for x in fresh if x in text]
                if hit:
                    return fail([f'{ms.group(1)}: quotes not-yet-seen result value(s) {hit[:3]}'])
            seen.append(text)
            cv = (m.get('content') or '').strip()
            mc = SENT.fullmatch(cv)
            seen.append(fills[mc.group(1)] if mc else cv)
        else:
            seen.append(m.get('content') or '')
    if check_only:
        print(f'{task}: fills OK ({len(slots)} slots)')
        return True

    for m in skel['messages']:
        if m['role'] != 'assistant':
            continue
        for field in ('reasoning_content', 'content'):
            v = (m.get(field) or '').strip()
            ms = SENT.fullmatch(v)
            if ms:
                m[field] = fills[ms.group(1)].strip()
    with open(f'{d}/agentic_v2.json', 'w', encoding='utf-8') as f:
        json.dump(skel, f, ensure_ascii=False, indent=1)
    r = subprocess.run([sys.executable, 'tools/build_agentic_v2.py', '--lint', task],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        return fail([f'{task}: post-apply lint failed'])
    print(f'{task}: applied {len(slots)} fills, lint clean')
    return True


def main():
    check = '--check' in sys.argv
    tasks = [a for a in sys.argv[1:] if not a.startswith('-')]
    if not tasks:
        print(__doc__)
        sys.exit(2)
    ok = all(apply_task(t, check_only=check) for t in tasks)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
