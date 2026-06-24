#!/usr/bin/env python3
"""Build the V4 batch: new FrontierCS-style competition-deliverable SFT traces -> ShareGPT jsonl.

Reads every data_v4/<slug>/ that has context.md + reasoning.md + train_answer.md (the verified
datapoints produced by tools/gen_v4_workflow.js + the hand-authored flagship) and emits
sft/innovation_v4_sft.jsonl in the SAME LLaMA-Factory ShareGPT format build_sft.py uses, so it can be
mixed straight into the SFT run (add it as a second dataset, or concatenate).

Design choices, matching the remediation plan (experiments/DATA_REMEDIATION_zh.md):
  - landing point is a single-file C++ solution reading stdin (the format FrontierCS judges), not a
    research-Python library;
  - the trace's spine is debug + self-verify of both reasoning and code (the gen workflow enforces it);
  - system prompt carries delivery + verification discipline, consistent with the eval framing
    ("solve in C++, output the code", judged single-shot).

Usage:  python sft/build_v4.py            # writes sft/innovation_v4_sft.jsonl + sft/_v4_tags.jsonl
"""
import json, os, glob, re

REPO = os.environ.get('INNOVATION_PRIOR_REPO') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

V4_SYS = ("You are an expert competitive programmer. Solve the problem with a single, self-contained "
          "C++ program that reads from standard input and writes to standard output. Before you commit "
          "to an answer, verify your reasoning and your code: trace the code on concrete inputs, check "
          "the edge cases, and fix any bug you find. Output the final solution as one C++ code block.")

_NOTE = ("\n\nThink it through in a flowing, first-person tone: weigh the candidate approaches, derive "
         "the one you can prove correct, write the code, then trace it on concrete inputs and the edge "
         "cases to debug it before committing. Finish on a single complete, self-contained C++ program "
         "that respects the input/output contract.")

_NEUT = {'<think>': '⟨think⟩', '</think>': '⟨/think⟩'}
def neutralize(s):
    for k, v in _NEUT.items():
        s = s.replace(k, v)
    return s
def read(p):
    return neutralize(open(p, encoding='utf-8').read().strip())
def think(reasoning, answer):
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer.strip()}"

_STDIN_RE = re.compile(r'\b(std::cin|cin\s*>>|scanf|getline)')
_CPP_RE = re.compile(r'```(?:cpp|c\+\+|cc)\b', re.I)
_DEBUG_RE = re.compile(r'(trace|the bug|off-by-one|overflow|edge case|re-?verif|counterexample|debug)', re.I)

examples, tags = [], []
for d in sorted(glob.glob('data_v4/*/')):
    slug = d.rstrip('/').split('/')[-1]
    files = {f: os.path.join(d, f'{f}.md') for f in ('context', 'reasoning', 'train_answer')}
    if not all(os.path.isfile(p) for p in files.values()):
        continue
    ctx = read(files['context'])
    rsn = read(files['reasoning'])
    ans = read(files['train_answer'])
    human = ctx + _NOTE
    gpt = think(rsn, ans)
    examples.append({'conversations': [{'from': 'human', 'value': human},
                                       {'from': 'gpt', 'value': gpt}],
                     'system': V4_SYS, '_id': slug})
    tags.append({'id': slug, 'reads_stdin': bool(_STDIN_RE.search(gpt)),
                 'has_cpp': bool(_CPP_RE.search(gpt)), 'reasoning_chars': len(rsn),
                 'has_debug_episode': bool(_DEBUG_RE.search(rsn))})

out = 'sft/innovation_v4_sft.jsonl'
with open(out, 'w', encoding='utf-8') as f:
    for ex in examples:
        f.write(json.dumps({k: v for k, v in ex.items() if not k.startswith('_')}, ensure_ascii=False) + "\n")
with open('sft/_v4_tags.jsonl', 'w', encoding='utf-8') as f:
    for t in tags:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

n = len(examples) or 1
import statistics
chars = [t['reasoning_chars'] for t in tags] or [0]
print(f"wrote {out}: {len(examples)} V4 examples")
print(f"  reads_stdin: {sum(t['reads_stdin'] for t in tags)} ({100*sum(t['reads_stdin'] for t in tags)/n:.0f}%)"
      f" | has_cpp: {sum(t['has_cpp'] for t in tags)} ({100*sum(t['has_cpp'] for t in tags)/n:.0f}%)"
      f" | has_debug_episode: {sum(t['has_debug_episode'] for t in tags)} ({100*sum(t['has_debug_episode'] for t in tags)/n:.0f}%)")
print(f"  reasoning chars: median {int(statistics.median(chars))}, min {min(chars)}, max {max(chars)}")
print("  add to LLaMA-Factory dataset_info.json as a second dataset, or concatenate into innovation_sft.jsonl")
