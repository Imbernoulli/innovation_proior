#!/usr/bin/env python3
"""Assemble the public Qwen-distilled pieces into ONE ShareGPT "maintain" SFT file.

This is the capability-MAINTENANCE set (`maintain_sft.jsonl`): public Qwen-distilled traces mixed
in to keep the base model's original abilities (prevent forgetting / stay on-policy), trained in a
SINGLE run alongside our annotated `innovation_sft.jsonl`. It uses the per-example metadata of the
LlamaFactory fork `feat/per-turn-loss-mask`:
  * per-turn `loss`            -> folding (used by innovation_sft, not here)
  * per-example `enable_thinking` -> lets reasoning and NO-reasoning examples coexist in one run.

`maintain_sft.jsonl` (one file):
  reasoning (Qwen): khazarai/qwen3.6-plus-high-reasoning-500x (250),
                    WithinUsAI/Qwen3.7_Max_Thinking_dataset_5K (250),
                    armand0e/qwen3.7-max-pi-traces (47, all non-empty sessions),
                    armand0e/qwen3.7-plus-claude-code (6, all non-empty sessions);
  reasoning (MiniMax): nvidia/Open-SWE-Traces openhands minimax_m25 (100);
  NO-reasoning (Qwen): nvidia/Open-SWE-Traces openhands qwen35_122b (250) -> each tagged
                    "enable_thinking": false (its empty think lands in the prompt, never the loss,
                    so the model never learns open-think -> immediate close-think).

Tool declarations: the agentic traces (armand, Open-SWE) carry tool *calls* but declare no tool
*schemas*. We reconstruct a minimal `tools` declaration per example from the observed calls
(name + argument keys) so the tools appear in the system prompt at the right place.
"""
import json, os, re

SRC = '/tmp/sr_build'
OUT = '/srv/home/bohanlyu/innovation_proior/sft'

# (stub, cap_or_None, enable_thinking_flag_or_None)
SOURCES = [
    ('distill_khazarai',       None, None),
    ('distill_withinusai',     None, None),
    ('distill_armand_pi',      None, None),
    ('distill_armand_claude',  None, None),
    ('distill_openswe_think',  100,  None),    # MiniMax reasoning, capped to 100
    ('distill_openswe_nothink', None, False),  # Qwen no-reasoning -> enable_thinking=False
]

USER = {'human', 'observation'}
ASST = {'gpt', 'function_call'}
_NEUT = {'<|im_start|>': '⟨im_start⟩', '<|im_end|>': '⟨im_end⟩'}
def neutralize_special(s):
    for k, v in _NEUT.items():
        s = s.replace(k, v)
    return s

def valid(ex):
    cv = ex.get('conversations')
    if not cv or len(cv) < 2:
        return False
    sides = []
    for c in cv:
        if c['from'] in USER:
            sides.append('U')
        elif c['from'] in ASST:
            sides.append('A')
        else:
            return False
        if not str(c.get('value', '')).strip():
            return False
    return sides[0] == 'U' and sides[-1] == 'A' and all(a != b for a, b in zip(sides, sides[1:]))

_TC_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
def synth_tools(convs):
    """Reconstruct a minimal tool declaration from the tool CALLS observed in the conversation."""
    seen = {}  # name -> {'union': set, 'inter': set}
    for c in convs:
        if c['from'] != 'function_call':
            continue
        for m in _TC_RE.findall(c['value']):
            try:
                tc = json.loads(m)
            except json.JSONDecodeError:
                continue
            name, args = tc.get('name'), tc.get('arguments', {})
            if not name or not isinstance(args, dict):
                continue
            keys = set(args.keys())
            if name not in seen:
                seen[name] = {'union': set(keys), 'inter': set(keys)}
            else:
                seen[name]['union'] |= keys
                seen[name]['inter'] &= keys
    if not seen:
        return None
    tools = [{
        "type": "function",
        "function": {
            "name": name,
            "description": f"The `{name}` tool (schema reconstructed from observed calls).",
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in sorted(k_['union'])},
                "required": sorted(k_['inter']),
            },
        },
    } for name, k_ in seen.items()]
    return json.dumps(tools, ensure_ascii=False)

def load(stub):
    p = os.path.join(SRC, stub + '.jsonl')
    if not os.path.isfile(p):
        print(f"  !! missing {p}")
        return []
    rows = []
    for ln in open(p, encoding='utf-8'):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return rows

merged = []
stats = []
for stub, cap, et in SOURCES:
    rows = load(stub)
    kept = 0
    for ex in rows:
        if cap is not None and kept >= cap:
            break
        if not valid(ex):
            continue
        for c in ex['conversations']:
            c['value'] = neutralize_special(str(c['value']))
        if ex.get('system'):
            ex['system'] = neutralize_special(str(ex['system']))
        # Uniform top-level schema (load-safety, 2026-07): every row carries the SAME keys, so the
        # arrow schema never depends on which rows land in the first block -- optional columns that
        # appear only later in the file crash the datasets load (observed with datasets 4.8.5).
        # Reasoning rows get an explicit enable_thinking=true (identical behaviour to the qwen3
        # template default, but immune to schema/None handling).
        out = {'conversations': ex['conversations'],
               'system': ex.get('system') or "",
               # tools: keep declared, else reconstruct from observed calls
               'tools': (ex.get('tools') or synth_tools(ex['conversations'])) or "",
               'enable_thinking': True if et is None else et}
        merged.append(out)
        kept += 1
    has_think = sum(1 for e in merged[-kept:] if any('<think>' in c['value'] for c in e['conversations'] if c['from'] in ASST)) if kept else 0
    with_tools = sum(1 for e in merged[-kept:] if 'tools' in e) if kept else 0
    stats.append((stub, kept, has_think, with_tools, et))

os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, 'maintain_sft.jsonl')
with open(path, 'w', encoding='utf-8') as f:
    for ex in merged:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"wrote {path}: {len(merged)} examples")
for stub, kept, ht, wt, et in stats:
    tag = "enable_thinking=false" if et is False else ("enable_thinking=true" if et is True else "default")
    print(f"  {stub:26} kept={kept:4} with<think>={ht:4} with_tools={wt:4} [{tag}]")
n_nothink = sum(1 for e in merged if e.get('enable_thinking') is False)
print(f"  -> reasoning examples: {len(merged)-n_nothink} | no-reasoning (enable_thinking=false): {n_nothink}")
