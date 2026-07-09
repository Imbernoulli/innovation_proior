#!/usr/bin/env python3
"""Assemble THIS WAVE's verified data into one LLaMA-Factory ShareGPT SFT jsonl.

Sources (all verified / kept this wave, none yet in the training mix):
  1. hard-CP rollout keepers  data_v4/_hardcp/traces/{code,math,reasoning,ifollow}.jsonl
       - Qwen3.6-27B on-policy rejection samples (keep unless it aced round 0, i.e. first_round_rate<1)
     + data_v4/_hardcp/traces/*.deepseek.jsonl
       - DeepSeek V4 Pro tier-2 solving the 27B's hard failures (keep every solve; math via flash judge)
  2. Codex FrontierCS-style C++ datapoints  data_v4/_fcs_codex/gen/<id>/{context,reasoning,answer}.md
       - `codex exec` (gpt-5.5) black-box, single-file C++/stdin landing

(cpv4b/v4 C++ datapoints and methods/ dirs are built by sft/build_v4.py and sft/build_sft.py.)

Output: sft/innovation_wave2_sft.jsonl (+ _wave2_tags.jsonl). gzip it to ship (sft/*.jsonl is gitignored).
"""
import json, os, glob, statistics

REPO = os.environ.get('INNOVATION_PRIOR_REPO') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
HARDCP = 'data_v4/_hardcp'

# Optional hardness filter: keep only rollout keepers whose round-0 pass rate (first_round_rate,
# None==0) is <= WAVE_ACC_MAX. Codex/v4 datapoints have no acc and are unaffected.
ACC_MAX = float(os.environ['WAVE_ACC_MAX']) if os.environ.get('WAVE_ACC_MAX') else None

CODE_SYS = ("You are an expert competitive programmer. Solve the problem with a single, self-contained "
            "C++17 program that reads from standard input and writes to standard output. Before you commit, "
            "verify your reasoning and trace your code on concrete inputs and edge cases, and fix any bug "
            "you find. Output the final solution as one C++ code block.")
MATH_SYS = ("You are an expert mathematician. Solve the problem. Think step by step, verify your work, then "
            "give the final answer on its own in \\boxed{}.")
FCS_SYS = CODE_SYS
DOMAIN_SYS = {'code': CODE_SYS, 'math': MATH_SYS, 'reasoning': None, 'ifollow': None}

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
    for dom in ('code', 'math', 'reasoning', 'ifollow'):
        for suf in ('worklist.jsonl', 'failed_worklist.jsonl'):
            path = f'{HARDCP}/{dom}/{suf}'
            if os.path.exists(path):
                for l in open(path):
                    if l.strip():
                        d = json.loads(l)
                        wl.setdefault(d['id'], d)
    return wl


def rollout_examples(wl):
    ex = []
    for dom in ('code', 'math', 'reasoning', 'ifollow'):
        for src, keep_all in ((f'{HARDCP}/traces/{dom}.jsonl', False),
                              (f'{HARDCP}/traces/{dom}.deepseek.jsonl', True),
                              (f'{HARDCP}/traces/{dom}.poe.jsonl', True)):
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
                # 27B on-policy: drop the ones it aced 4/4 in round 0 (too easy); keep tier-2 solves all
                if not keep_all and (r.get('first_round_rate') or 0) >= 1.0:
                    continue
                # optional hardness cap: keep only round-0 acc <= ACC_MAX (None==0, i.e. escalation-only)
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
                          '_reasoning_chars': len(reasoning)})
                ex.append(e)
    return ex


def codex_examples():
    ex = []
    for d in sorted(glob.glob('data_v4/_fcs_codex/gen/*/')):
        mf = os.path.join(d, 'meta.json')
        if not os.path.exists(mf):
            continue
        m = json.load(open(mf))
        if not m.get('ok'):
            continue
        cat = m.get('category', 'algorithm')
        ctx = open(os.path.join(d, 'context.md')).read().strip()
        rsn = open(os.path.join(d, 'reasoning.md')).read().strip()
        ans = open(os.path.join(d, 'answer.md')).read().strip()
        if not (ctx and rsn and ans):
            continue
        conv = [{'from': 'human', 'value': ctx},
                {'from': 'gpt', 'value': think(rsn, ans)}]
        ex.append({'conversations': conv, 'system': FCS_SYS,
                   '_id': m['id'], '_domain': f'fcs_codex:{cat}', '_source': 'codex:gpt-5.5',
                   '_reasoning_chars': len(rsn)})
    return ex


def main():
    wl = load_worklists()
    ex = rollout_examples(wl) + codex_examples()
    # de-dup by (id, domain) keeping first
    seen, uniq = set(), []
    for e in ex:
        k = (e['_id'], e['_domain'])
        if k in seen:
            continue
        seen.add(k); uniq.append(e)
    out = 'sft/innovation_wave2_sft.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for e in uniq:
            f.write(json.dumps({k: v for k, v in e.items() if not k.startswith('_')},
                               ensure_ascii=False) + '\n')
    with open('sft/_wave2_tags.jsonl', 'w', encoding='utf-8') as f:
        for e in uniq:
            f.write(json.dumps({'id': e['_id'], 'domain': e['_domain'], 'source': e['_source'],
                                'reasoning_chars': e['_reasoning_chars']}, ensure_ascii=False) + '\n')
    from collections import Counter
    by_dom = Counter(e['_domain'] for e in uniq)
    chars = [e['_reasoning_chars'] for e in uniq] or [0]
    print(f'wrote {out}: {len(uniq)} examples')
    for dom, n in sorted(by_dom.items()):
        print(f'  {dom:20} {n}')
    print(f'  reasoning chars: median {int(statistics.median(chars))}, max {max(chars)}')


if __name__ == '__main__':
    main()
