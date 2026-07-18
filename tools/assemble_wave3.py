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

# Hard-only by default: keep a rollout keeper only if its round-0 pass rate (first_round_rate,
# None==0) is <= 0.5 — same "keep only hard samples" bar as wave-2. Override via WAVE_ACC_MAX
# (e.g. WAVE_ACC_MAX=1.0 to keep everything the 27B didn't ace 4/4).
ACC_MAX = float(os.environ['WAVE_ACC_MAX']) if os.environ.get('WAVE_ACC_MAX') else 0.5

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
    return [(f'{HARDCP}/traces/{dom}.jsonl', False),
            (f'{HARDCP}/traces/{dom}.ccplus.jsonl', False),   # code: dedicated CodeContests+ pass (on-policy)
            (f'{HARDCP}/traces/{dom}.deepseek.jsonl', True),
            (f'{HARDCP}/traces/{dom}.poe.jsonl', True),
            (f'{HARDCP}/traces/{dom}.reroll.jsonl', True)]

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
    wl = {}
    for dom in DOMAINS:
        for suf in ('worklist.jsonl', 'failed_worklist.jsonl', 'failed_hard.jsonl'):
            path = f'{HARDCP}/{dom}/{suf}'
            if os.path.exists(path):
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
                if not keep_all and (r.get('first_round_rate') or 0) >= 1.0:
                    continue
                if ACC_MAX is not None and (r.get('first_round_rate') or 0) > ACC_MAX:
                    continue
                p = wl.get(r['id'])
                if not p:
                    continue
                stmt = statement(p)
                pas = r['passes'][0]  # shortest kept generation
                reasoning, answer = pas.get('reasoning') or '', pas.get('answer') or ''
                if not stmt or not answer.strip():
                    continue
                conv = [{'from': 'human', 'value': stmt.strip()},
                        {'from': 'gpt', 'value': think(reasoning, answer)}]
                e = {'conversations': conv}
                if DOMAIN_SYS.get(dom):
                    e['system'] = DOMAIN_SYS[dom]
                e.update({'_id': r['id'], '_domain': dom, '_source': r.get('source', ''),
                          '_reasoning_chars': len(reasoning),
                          '_reroll': src.endswith('.reroll.jsonl')})
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
                                'reasoning_chars': e['_reasoning_chars'], 'reroll': e['_reroll']},
                               ensure_ascii=False) + '\n')
    by_dom = Counter(e['_domain'] for e in uniq)
    reroll_n = sum(1 for e in uniq if e['_reroll'])
    chars = [e['_reasoning_chars'] for e in uniq] or [0]
    print(f'wrote {out}: {len(uniq)} examples ({dropped_ship} skipped as already in wave-2)')
    for dom, n in sorted(by_dom.items()):
        print(f'  {dom:20} {n}')
    print(f'  of which deep-reroll keepers: {reroll_n}')
    print(f'  reasoning chars: median {int(statistics.median(chars))}, max {max(chars)}')


if __name__ == '__main__':
    main()
