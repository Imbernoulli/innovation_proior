#!/usr/bin/env python3
"""Real Qwen3.5 token length of every agentic v2 episode (the whole conversation as the
longest folded row would carry it). Writes tools/agentic_v2_toklen.json for build_sft's
cutoff guard. Run with the LF venv python (needs transformers):
  /srv/home/bohanlyu/LF-innov/.venv/bin/python tools/agentic_v2_toklen.py
"""
import glob, json, os
from transformers import AutoTokenizer
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# della has no internet on compute nodes and no 3.5-4B in cache; point this at the
# local 9B we actually train (same Qwen3.5 tokenizer) via AGENTIC_TOKLEN_MODEL.
tok = AutoTokenizer.from_pretrained(
    os.environ.get('AGENTIC_TOKLEN_MODEL', 'Qwen/Qwen3.5-4B'), trust_remote_code=True)
out = {}
for p in sorted(glob.glob('trajectories/*/agentic_v2_filled.json')):
    d = json.load(open(p, encoding='utf-8'))
    text = d['system'] + json.dumps(d['tools']) + ''.join(
        (m.get('content') or '') + (m.get('reasoning_content') or '') + json.dumps(m.get('tool_calls') or [])
        for m in d['messages'])
    out[d['task']] = len(tok(text)['input_ids'])
json.dump(out, open('tools/agentic_v2_toklen.json', 'w'), indent=1, sort_keys=True)
big = sorted(out.items(), key=lambda x: -x[1])[:8]
print('episodes', len(out), '| longest:', big)
