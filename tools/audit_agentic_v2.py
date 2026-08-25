#!/usr/bin/env python3
"""Multi-angle audit of the agentic v2 corpus (trajectories/*/agentic_v2_filled.json).

Deterministic checks, each printing per-task findings and a summary line:
  A. structure   : replay integrity, sentinels, empty/oversized thinks (lint)
  B. plan-vs-edit: rewrite/create turns whose SAY/design text narrates piecewise edits
  C. numbers     : (i) FUTURE LEAK — a metric-looking number that first appears in a LATER
                   tool result is quoted earlier; (ii) UNGROUNDED — a >=4-significant-digit
                   number never visible anywhere in the transcript
  D. voice       : out-of-frame vocabulary (rung/ladder/arXiv/et al./repo paths/.md/slots)
  E. boilerplate : near-identical thinks of the same kind inside one task
  F. copying     : 8-gram overlap of design thinks vs the rung reasoning files
  G. contract    : system prompt + tool schemas byte-equal to the MLS-Bench checkout defaults
  H. cumulative  : CUMULATIVE_TASKS episodes are create-only; others replay to target
Exit 1 if any HARD finding (A, C-i, G, H) exists; B/C-ii/D/E/F are reported for review.

Usage: python3 tools/audit_agentic_v2.py [TASK ...]
"""
import difflib, glob, json, os, re, sys, importlib.util
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
spec = importlib.util.spec_from_file_location('b', 'tools/build_agentic_v2.py')
B = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]; spec.loader.exec_module(B)

TELL = re.compile(r"\b(first (edit|hunk|patch)|this (edit|hunk|patch) (renames|reformats|strips|touches|only)|next (edit|hunk)|ahead of the (swap|rewrite)|hunks?|surgical|bottom-up|top-down|second edit|remaining edits|the edits? (below|that follow)|str_replace|old_str|new_str)\b", re.I)
OOF = re.compile(r"(\brungs?\b|\bladder\b|\barxiv\b|\bet al\.|\bsentinel\b|\[\[|\]\]|results/[a-z]|methods/[a-z]|trajectories/[a-z]|reasoning\.md|answer\.md|context\.md|\bthe paper\b|this transcript|generation process)", re.I)
NUM = re.compile(r'(?<![\w.])(\d+\.\d+|\d{3,})(?![\w.])')
ALLOW_CI = {(e['task'], e['msg'], e['number']) for e in json.load(open('tools/agentic_v2_audit_allow.json'))['Ci']} if os.path.isfile('tools/agentic_v2_audit_allow.json') else set()
def sig(n): return len(n.replace('.', '').lstrip('0'))
def canon(vals):
    out = set()
    for v in vals:
        try: f = float(v)
        except ValueError: continue
        d = len(v.split('.')[1]) if '.' in v else 0
        for k in range(0, d + 1):
            out.add(f'{round(f, k):.{k}f}')
    return out
def known(n, pool):
    return n in pool or (('.' in n) and f'{float(n):.{len(n.split(".")[1])}f}' in pool)

def audit(task):
    d = json.load(open(f'trajectories/{task}/agentic_v2_filled.json'))
    ms = d['messages']; F = {'B': [], 'Ci': [], 'Cii': [], 'D': [], 'E': [], 'F': []}
    cumulative = d.get('cumulative_stack')
    # --- C: numbers. visible-so-far vs later tool results
    tool_texts = [(i, m['content']) for i, m in enumerate(ms) if m['role'] == 'tool']
    seen = canon(NUM.findall(ms[0]['content']))
    all_visible = set(seen)
    for i, m in enumerate(ms):
        if m['role'] == 'tool': all_visible |= canon(NUM.findall(m['content']))
        elif m['role'] == 'assistant':
            all_visible |= canon(NUM.findall(' '.join(str(v) for v in m['tool_calls'][0]['function']['arguments'].values())))
    kinds = {}
    for i, m in enumerate(ms):
        if m['role'] == 'tool':
            seen |= canon(NUM.findall(m['content'])); continue
        if m['role'] != 'assistant': continue
        rc = m.get('reasoning_content') or ''; say = m.get('content') or ''
        fn = m['tool_calls'][0]['function']; args = fn['arguments']
        op = args.get('op') if fn['name'] == 'edit' else fn['name']
        # numbers the engineer writes into the code this turn are theirs, not leaks/ungrounded
        code_nums = canon(NUM.findall(' '.join(str(v) for v in args.values())))
        seen |= code_nums; all_visible |= code_nums
        # numbers in prose (exclude code args)
        nums = [n for n in NUM.findall(rc + ' ' + say) if sig(n) >= 4]
        later = set()
        for j, t in tool_texts:
            if j > i: later |= canon(NUM.findall(t))
        for n in set(nums):
            if known(n, seen): continue
            if (task, i, n) in ALLOW_CI: continue
            if known(n, later): F['Ci'].append((i, n))
            elif not known(n, all_visible): F['Cii'].append((i, n))
        # --- B
        if op in ('rewrite', 'create'):
            for field, txt in (('think', rc), ('say', say)):
                hits = sorted(set(h[0] for h in TELL.findall(txt)))
                if hits: F['B'].append((i, field, hits[:3]))
        # --- D
        for field, txt in (('think', rc), ('say', say)):
            oh = sorted(set(x.lower() for x in OOF.findall(txt)))
            if oh: F['D'].append((i, field, oh[:4]))
        # --- E buckets
        k = 'test' if fn['name'] == 'test' else ('submit' if fn['name'] == 'submit' else 'edit')
        kinds.setdefault(k, []).append((i, rc))
        seen |= canon(NUM.findall(rc + ' ' + say))
    # --- E: near-identical thinks
    for k, lst in kinds.items():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                ra, rb = lst[a][1], lst[b][1]
                if not (80 < len(ra) < 1500 and 80 < len(rb) < 1500): continue   # short thinks only
                if abs(len(ra) - len(rb)) > 0.3 * max(len(ra), len(rb)): continue
                if difflib.SequenceMatcher(a=ra, b=rb).quick_ratio() > 0.85 \
                   and difflib.SequenceMatcher(a=ra, b=rb).ratio() > 0.8:
                    F['E'].append((lst[a][0], lst[b][0], k))
    # --- F: copying
    meta = json.load(open(f'trajectories/{task}/meta.json'))
    ref = ''
    for s in meta['steps']:
        rp = f"trajectories/{task}/{s.get('reasoning', '')}"
        if s.get('reasoning') and os.path.isfile(rp): ref += open(rp).read() + '\n'
    def ng(s): w = re.findall(r'\w+', s.lower()); return set(tuple(w[i:i + 8]) for i in range(len(w) - 7))
    rg = ng(ref)
    for i, m in enumerate(ms):
        if m['role'] == 'assistant' and m['tool_calls'][0]['function']['name'] == 'edit':
            g = ng(m.get('reasoning_content') or '')
            if g and len(g & rg) / len(g) > 0.02: F['F'].append((i, round(100 * len(g & rg) / len(g), 1)))
    # --- H
    H = []
    for m in ms:
        if m['role'] == 'assistant' and m['tool_calls'][0]['function']['name'] == 'edit':
            op = m['tool_calls'][0]['function']['arguments']['op']
            if cumulative and op != 'create': H.append(op)
    return F, H

def contract_check():
    root = 'training/FrontierSmith/.cache/mlsbench-eval/src'
    sys.path.insert(0, root)
    for k in ('MLSBENCH_STRICT_STR_REPLACE',): os.environ.pop(k, None)
    from mlsbench.agent import interactive as I
    from mlsbench.agent.tools import TOOL_SCHEMAS, EDIT_REWRITE_SCHEMA, VIEW_SCHEMA
    src = open(f'{root}/mlsbench/agent/interactive.py').read()
    # reproduce __init__ (use_replace, rewrite on, view on): pull the literal blocks from source
    def lit(start_marker, end_marker):
        i = src.index(start_marker); j = src.index(end_marker, i)
        block = src[i:j]
        return ''.join(re.findall(r'"((?:[^"\\]|\\.)*)"', block)).replace('\\n', '\n').replace("\\'", "'")
    edit_block = lit('edit_block = (', ')\n')
    rb = lit('elif _rewrite_enabled:', 'if _view_enabled:')
    rb += lit('if _view_enabled:\n                    replace_block += (', ')\n                replace_block += (')
    rb += lit('replace_block += (\n                    "\\n"', ')\n            else:')
    sp = I.SYSTEM_PROMPT_SCI.replace(edit_block, rb, 1)
    problems = []
    if sp != B.SYSTEM_PROMPT: problems.append('SYSTEM_PROMPT differs from checkout composition')
    want = {'edit': EDIT_REWRITE_SCHEMA, 'view': VIEW_SCHEMA}
    for s in TOOL_SCHEMAS:
        if s['name'] != 'edit': want[s['name']] = s
    for t in B.TOOLS:
        f = t['function']; w = want.get(f['name'])
        if not w: problems.append(f"extra tool {f['name']}"); continue
        if f['description'] != w['description']: problems.append(f"{f['name']}: description differs")
        if f['parameters'] != w['input_schema']: problems.append(f"{f['name']}: parameters differ")
    return problems

def main():
    tasks = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob('trajectories/*/agentic_v2_filled.json'))
    import subprocess
    lint = subprocess.run([sys.executable, 'tools/build_agentic_v2.py', '--lint'] + tasks, capture_output=True, text=True)
    hard = 0
    print('A structure:', lint.stdout.strip().splitlines()[-1]); hard += lint.returncode != 0
    tot = {k: 0 for k in 'B Ci Cii D E F H'.split()}; per = {}
    for t in tasks:
        F, H = audit(t)
        for k, v in F.items(): tot[k] += len(v)
        tot['H'] += len(H)
        if any(F.values()) or H: per[t] = (F, H)
    print(f"B plan-vs-edit: {tot['B']}   C-i FUTURE-LEAK: {tot['Ci']}   C-ii ungrounded: {tot['Cii']}   D out-of-frame: {tot['D']}   E boilerplate: {tot['E']}   F copying>2%: {tot['F']}   H cumulative-op: {tot['H']}")
    hard += tot['Ci'] + tot['H']
    try:
        cp = contract_check(); print('G contract:', 'OK' if not cp else cp); hard += len(cp)
    except Exception as e:
        print('G contract: check failed to run:', e); hard += 1
    out = {t: {'B': F['B'], 'Ci': F['Ci'], 'Cii': F['Cii'], 'D': F['D'], 'E': F['E'], 'F': F['F'], 'H': H} for t, (F, H) in per.items()}
    json.dump(out, open('tools/agentic_v2_audit.json', 'w'), indent=1)
    print(f"details -> tools/agentic_v2_audit.json ({len(per)} tasks with findings); HARD findings: {hard}")
    sys.exit(1 if hard else 0)

if __name__ == '__main__':
    main()
