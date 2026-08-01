#!/usr/bin/env python3
"""Assemble WAVE-3: every verified keeper NOT already shipped in wave-2, as one
LLaMA-Factory ShareGPT SFT jsonl.

Wave-3 = (all current hard-CP keepers, current verifier logic) MINUS (wave-2 ids).
It therefore picks up, with no overlap vs wave-2:
  - NEW capability tracks   data_v4/_hardcp/traces/{optim,ahc}.jsonl
  - code growth since wave2  data_v4/_hardcp/traces/code.jsonl (ccplus + first-pass continuation)
  - math/reasoning/ifollow growth in the base traces
  - DEEP re-roll of the 27B's hard-failures  traces/{math,reasoning,ifollow}.reroll.jsonl (keep every solve)
  - tier-2 teacher solves  traces/*.deepseek.jsonl / *.poe.jsonl (keep every solve)

On-policy base traces: drop the ones the 27B aced 4/4 in round 0 (too easy). Teacher /
reroll passes are all genuine hard-failures -> keep every solve. De-dup by (id, domain),
then subtract anything already in sft/_wave2_tags.jsonl.

Output: sft/innovation_wave3_sft.jsonl (+ _wave3_tags.jsonl). gzip to ship (sft/*.jsonl is gitignored).
"""
import json, os, glob, statistics
from collections import Counter

REPO = os.environ.get('INNOVATION_PRIOR_REPO') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
HARDCP = 'data_v4/_hardcp'

# Policy (2026-08): ship EVERY query that has >=1 verified-correct generation — one answer per
# query — and LABEL each with its round-0 pass rate (`pass_rate`). NO accuracy cap by default:
# we no longer keep only the hard (acc<=0.5) slice; downstream can filter on the labeled pass_rate
# instead. Set WAVE_ACC_MAX to re-impose a cap (e.g. WAVE_ACC_MAX=0.5 for the old hard-only cut).
ACC_MAX = float(os.environ['WAVE_ACC_MAX']) if os.environ.get('WAVE_ACC_MAX') else None

CODE_SYS = ("You are an expert competitive programmer. Solve the problem with a single, self-contained "
            "C++17 program that reads from standard input and writes to standard output. Before you commit, "
            "verify your reasoning and trace your code on concrete inputs and edge cases, and fix any bug "
            "you find. Output the final solution as one C++ code block.")
MATH_SYS = ("You are an expert mathematician. Solve the problem. Think step by step, verify your work, then "
            "give the final answer on its own in \\boxed{}.")
# optim/ahc are single-file C++ reading stdin (heuristic optimization) -> same contract as code.
DOMAIN_SYS = {'code': CODE_SYS, 'math': MATH_SYS, 'reasoning': None, 'ifollow': None,
              'optim': CODE_SYS, 'ahc': CODE_SYS}

# (base trace, keep_all?) per domain. keep_all=True => keep every solve (teacher / reroll / hardest).
DOMAINS = ['code', 'math', 'reasoning', 'ifollow', 'optim', 'ahc']
def sources_for(dom):
    # Order matters: de-dup keeps the FIRST passing record per (id, domain), so the modern
    # escalation traces win; the archived phases (.oldlogic stop-at-first-pass, .hardv2/.mixed/
    # .hardrun old math runs, .measure) only fill queries nothing else solved.
    return [(f'{HARDCP}/traces/{dom}.jsonl', False),
            (f'{HARDCP}/traces/{dom}.ccplus.jsonl', False),   # code: dedicated CodeContests+ pass (on-policy)
            (f'{HARDCP}/traces/{dom}.deepseek.jsonl', True),
            (f'{HARDCP}/traces/{dom}.poe.jsonl', True),
            (f'{HARDCP}/traces/{dom}.reroll.jsonl', True),
            (f'{HARDCP}/traces/{dom}.hardv2.jsonl', True),
            (f'{HARDCP}/traces/{dom}.mixed.jsonl', True),
            (f'{HARDCP}/traces/{dom}.hardrun.jsonl', True),
            (f'{HARDCP}/traces/{dom}.oldlogic.jsonl', True),
            (f'{HARDCP}/traces/{dom}.measure.jsonl', True)]

_NEUT = {'<think>': '⟨think⟩', '</think>': '⟨/think⟩'}
def neutralize(s):
    for k, v in _NEUT.items():
        s = s.replace(k, v)
    return s
def think(reasoning, answer):
    return f"<think>\n{neutralize(reasoning.strip())}\n</think>\n\n{neutralize(answer.strip())}"
def statement(p):
    return p.get('statement') or p.get('question') or p.get('problem') or p.get('prompt')


def load_worklists():
    # Glob EVERY worklist/failed file per domain (worklist.jsonl, worklist_rstar.jsonl,
    # worklist_hardtests.jsonl, failed_*.jsonl, ...) — archived traces reference ids from
    # retired worklists (e.g. code.oldlogic's rstar_*), which the fixed three-name list missed.
    wl = {}
    for dom in DOMAINS:
        for path in sorted(glob.glob(f'{HARDCP}/{dom}/worklist*.jsonl') +
                           glob.glob(f'{HARDCP}/{dom}/failed*.jsonl')):
            for l in open(path):
                if l.strip():
                    d = json.loads(l)
                    wl.setdefault(d['id'], d)
    return wl


def shipped_wave2():
    seen = set()
    p = 'sft/_wave2_tags.jsonl'
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                d = json.loads(l)
                seen.add((d['id'], d['domain']))
    return seen


def rollout_examples(wl):
    ex = []
    for dom in DOMAINS:
        for src, keep_all in sources_for(dom):
            if not os.path.exists(src):
                continue
            for l in open(src):
                if not l.strip():
                    continue
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if not r.get('passed') or not r.get('passes'):
                    continue
                # Keep every query with a verified-correct generation; optional cap only if set.
                frr = r.get('first_round_rate')
                if ACC_MAX is not None and (frr or 0) > ACC_MAX:
                    continue
                p = wl.get(r['id'])
                if not p:
                    continue
                stmt = statement(p)
                pas = r['passes'][0]  # shortest kept generation
                reasoning, answer = pas.get('reasoning') or '', pas.get('answer') or ''
                if not stmt or not answer.strip():
                    continue
                # round-0 pass rate for THIS query (npass/valid over the first round's samples).
                # Always a float -> uniform schema (no loss:null-style landmine). -1.0 = UNKNOWN:
                # the archived stop-at-first-pass phase (.oldlogic etc.) recorded no round-0 batch,
                # so no rate exists; do NOT fake 0.0 ("hardest") for those.
                pass_rate = round(frr, 3) if isinstance(frr, (int, float)) else -1.0
                conv = [{'from': 'human', 'value': stmt.strip()},
                        {'from': 'gpt', 'value': think(reasoning, answer)}]
                e = {'conversations': conv, 'pass_rate': pass_rate}
                if DOMAIN_SYS.get(dom):
                    e['system'] = DOMAIN_SYS[dom]
                e.update({'_id': r['id'], '_domain': dom, '_source': r.get('source', ''),
                          '_reasoning_chars': len(reasoning),
                          '_reroll': src.endswith('.reroll.jsonl'),
                          '_pass_rate': pass_rate})
                ex.append(e)
    return ex


def main():
    wl = load_worklists()
    ex = rollout_examples(wl)
    wave2 = shipped_wave2()
    # de-dup by (id, domain) keeping first; then subtract wave-2
    seen, uniq, dropped_ship = set(), [], 0
    for e in ex:
        k = (e['_id'], e['_domain'])
        if k in seen:
            continue
        seen.add(k)
        if k in wave2:
            dropped_ship += 1
            continue
        uniq.append(e)
    out = 'sft/innovation_wave3_sft.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for e in uniq:
            f.write(json.dumps({k: v for k, v in e.items() if not k.startswith('_')},
                               ensure_ascii=False) + '\n')
    with open('sft/_wave3_tags.jsonl', 'w', encoding='utf-8') as f:
        for e in uniq:
            f.write(json.dumps({'id': e['_id'], 'domain': e['_domain'], 'source': e['_source'],
                                'reasoning_chars': e['_reasoning_chars'], 'reroll': e['_reroll'],
                                'pass_rate': e['_pass_rate']},
                               ensure_ascii=False) + '\n')
    by_dom = Counter(e['_domain'] for e in uniq)
    reroll_n = sum(1 for e in uniq if e['_reroll'])
    chars = [e['_reasoning_chars'] for e in uniq] or [0]
    pr_bucket = Counter(e['_pass_rate'] for e in uniq)
    hard = sum(1 for e in uniq if 0 <= e['_pass_rate'] <= 0.5)   # -1.0 = unknown, not "hard"
    print(f'wrote {out}: {len(uniq)} examples ({dropped_ship} skipped as already in wave-2)')
    for dom, n in sorted(by_dom.items()):
        print(f'  {dom:20} {n}')
    print(f'  of which deep-reroll keepers: {reroll_n}')
    print(f'  reasoning chars: median {int(statistics.median(chars))}, max {max(chars)}')
    print(f'  pass_rate distribution (round-0 acc): ' +
          ', '.join(f'{k}:{pr_bucket[k]}' for k in sorted(pr_bucket)))
    print(f'  hard (pass_rate<=0.5): {hard} / {len(uniq)}')


if __name__ == '__main__':
    main()
