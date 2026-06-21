#!/usr/bin/env python3
"""Build the LLaMA-Factory ShareGPT SFT data as NATURAL multi-turn conversations.

No invented structure: every assistant turn stays an assistant turn carrying its real
`<think>`; tool results stay `observation`; the test feedback stays a real turn. The two
input distributions the model must learn (history-with-reasoning vs history-stripped) are
produced by LlamaFactory's built-in `mask_history` flag + the official chat template, NOT
by reshaping the data:

  * The Qwen3/Qwen3.5 template renders a STRIPPED history turn as the bare assistant answer
    with the `<think>...</think>` removed (no block at all). LlamaFactory's `mask_history=True`
    reproduces exactly this (remove_thought on history) AND computes loss only on the last turn.
  * `mask_history=False` keeps every turn's `<think>` and trains every turn (the tool-loop /
    observation distribution, where reasoning is retained).

`mask_history` is a global training flag, so the two distributions are two files:

  KEPT  (sft/innovation_sft_kept.jsonl)      -> train with mask_history=False
        (a) methods            single-turn Q&A
        (b) trajectories        full multi-turn; feedback = observation
        (c) agentic             full tool loop; ALL results (incl. run_experiment) = observation;
                                assistant steps = structured function_call (qwen3 JSON / qwen3_5 XML)
      => every reasoning / answer / tool-call trained, history reasoning retained.

  STRIPPED (sft/innovation_sft_stripped.jsonl) -> train with mask_history=True
        Same conversations, but emitted as one PREFIX per target assistant turn (truncated
        there), with real roles and real <think> left in place -- LF strips the history think
        and trains only that last (target) turn. Trajectory feedback and agentic run_experiment
        results become `user` turns (the new-query boundary); str_replace results stay observation.
      => every target turn trained once in a history-reasoning-stripped context.

System prompt carries the discovery YEAR (method year for (a); trajectory first-method year
for (b)/(c)) as meta-conditioning.
"""
import json, os, glob

REPO = '/srv/home/bohanlyu/innovation_proior'
os.chdir(REPO)

METHOD_SYS = ("You are a machine learning researcher working in the year {year}. "
             "Given a research problem and the prior art available at that time, derive a method "
             "from first principles, reasoning through the design, and present the resulting method "
             "with working code.")
TRAJ_SYS = ("You are a machine learning researcher working in the year {year}. You iteratively "
           "design and improve a method for a fixed task over several rounds; after each round you "
           "are shown the measured results and must reason about them and refine your approach.")
AGENT_SYS = ("You are a research engineer improving a model on a fixed ML task. You work by "
            "editing the model code and running experiments. Think carefully before each change. "
            "Make one tool call at a time. After editing, run an experiment to measure your method. "
            "The current year is {year}.")

_NEUT = {'<think>':'⟨think⟩','</think>':'⟨/think⟩','<tool_call>':'⟨tool_call⟩',
         '</tool_call>':'⟨/tool_call⟩','<tool_response>':'⟨tool_response⟩',
         '</tool_response>':'⟨/tool_response⟩','<|im_start|>':'⟨im_start⟩','<|im_end|>':'⟨im_end⟩'}
def neutralize(s):
    for k, v in _NEUT.items():
        s = s.replace(k, v)
    return s
def read(p):
    return neutralize(open(p, encoding='utf-8').read().strip())
def think(reasoning, answer):
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer.strip()}"

kept = []        # mask_history=False
stripped = []    # mask_history=True (one prefix per target turn)
stats = {'method':0,'traj_kept':0,'traj_stripped':0,'agentic_kept':0,'agentic_stripped':0}

def end_on_assistant(convs):
    while convs and convs[-1]['from'] in ('human', 'observation'):
        convs.pop()
    return convs

def emit_prefixes(convs, sys, dst, kind, base_id, tools=None):
    """One STRIPPED sample per assistant-side target turn: the conversation truncated there,
    real roles + real <think> preserved (LF's mask_history=True strips history think and trains
    only this last turn). `user`/`observation` history turns just ride along as context."""
    n = 0
    for t in range(len(convs)):
        if convs[t]['from'] not in ('gpt', 'function_call'):
            continue
        pref = [dict(c) for c in convs[:t + 1]]
        if len(pref) < 2:
            continue
        ex = {'conversations': pref, 'system': sys, '_kind': kind, '_id': f'{base_id}#t{t}'}
        if tools is not None:
            ex['tools'] = tools
        dst.append(ex); n += 1
    return n

# ---------- (a) methods : single-turn (KEPT only) ----------
methods = json.load(open('methods.json'))
for m in methods:
    slug, yr = m['slug'], m.get('year')
    d = f'methods/{slug}/results'
    if not all(os.path.isfile(f'{d}/{f}.md') for f in ('context', 'reasoning', 'train_answer')):
        continue
    kept.append({'conversations': [{'from':'human','value':read(f'{d}/context.md')},
                                    {'from':'gpt','value':think(read(f'{d}/reasoning.md'), read(f'{d}/train_answer.md'))}],
                 'system': METHOD_SYS.format(year=yr), '_kind':'method', '_id':slug})
    stats['method'] += 1

# ---------- (b) trajectories ----------
trajs = {x['task']: x for x in json.load(open('trajectories.json'))}
def step_answer(d, st):
    rsn_f = st.get('reasoning')
    ta = os.path.join(d, os.path.basename(rsn_f).replace('-reasoning.md', '-train_answer.md'))
    if os.path.isfile(ta):
        return read(ta)
    ans_f = st.get('answer')
    return read(os.path.join(d, ans_f)) if ans_f and os.path.isfile(os.path.join(d, ans_f)) else ''

for meta_p in sorted(glob.glob('trajectories/*/meta.json')):
    task = os.path.basename(os.path.dirname(meta_p))
    tj = trajs.get(task); yr = tj['year'] if tj else None
    d = f'trajectories/{task}'
    meta = json.load(open(meta_p))
    steps = [s for s in sorted(meta.get('steps', []), key=lambda s: s.get('n', 0)) if s.get('reasoning')]
    if not steps:
        continue
    init = read(f"{d}/{meta.get('initial_context_file','00-initial-context.md')}")
    sysp = TRAJ_SYS.format(year=yr)

    # KEPT: feedback as observation
    convs = [{'from':'human','value':init}]
    for st in steps:
        convs.append({'from':'gpt','value':think(read(f"{d}/{st['reasoning']}"), step_answer(d, st))})
        fb = st.get('feedback')
        if fb and os.path.isfile(f"{d}/{fb}"):
            convs.append({'from':'observation','value':read(f"{d}/{fb}")})
    convs = end_on_assistant(convs)
    if len(convs) >= 2:
        kept.append({'conversations':convs, 'system':sysp, '_kind':'traj_kept','_id':task})
        stats['traj_kept'] += 1

    # STRIPPED: feedback as a `user` boundary; one prefix per rung (LF strips history think)
    sconvs = [{'from':'human','value':init}]
    for si, st in enumerate(steps):
        sconvs.append({'from':'gpt','value':think(read(f"{d}/{st['reasoning']}"), step_answer(d, st))})
        fb = st.get('feedback')
        if fb and os.path.isfile(f"{d}/{fb}") and si < len(steps) - 1:
            sconvs.append({'from':'human','value':read(f"{d}/{fb}")})
    stats['traj_stripped'] += emit_prefixes(sconvs, sysp, stripped, 'traj_stripped', task)

# ---------- (c) agentic ----------
def fc_value(msg):
    parts = []
    rc = (msg.get('reasoning_content') or '').strip()
    if rc:
        parts.append(f"<think>\n{neutralize(rc)}\n</think>")
    ct = (msg.get('content') or '').strip()
    if ct:
        parts.append(neutralize(ct))
    fn = msg['tool_calls'][0]['function']
    call = json.dumps({'name': fn['name'], 'arguments': fn['arguments']}, ensure_ascii=False)
    parts.append(f"<tool_call>\n{call}\n</tool_call>")
    return "\n\n".join(parts)

for ap in sorted(glob.glob('trajectories/*/agentic_messages.json')):
    task = os.path.basename(os.path.dirname(ap))
    yr = trajs[task]['year'] if task in trajs else None
    data = json.load(open(ap))
    tools_str = json.dumps(data.get('tools', []), ensure_ascii=False)
    sysp = AGENT_SYS.format(year=yr)

    # walk once; build KEPT (all results=observation) and STRIPPED (run_experiment result=user)
    kept_c = []; strip_c = []; last_call = None
    have_user = False
    for msg in data['messages']:
        r = msg['role']
        if r == 'system':
            continue
        if r == 'user':
            v = neutralize(msg['content']); kept_c.append({'from':'human','value':v}); strip_c.append({'from':'human','value':v}); have_user = True
        elif r == 'assistant':
            v = {'from':'function_call','value':fc_value(msg)} if msg.get('tool_calls') else None
            if v is None:
                txt = ((f"<think>\n{neutralize(msg['reasoning_content'].strip())}\n</think>\n\n" if (msg.get('reasoning_content') or '').strip() else '')
                       + neutralize((msg.get('content') or '').strip()))
                if not txt.strip():
                    continue
                v = {'from':'gpt','value':txt}
            kept_c.append(dict(v)); strip_c.append(dict(v))
            last_call = (msg['tool_calls'][0]['function']['name'] if msg.get('tool_calls') else None)
        elif r == 'tool':
            v = neutralize(msg['content'])
            kept_c.append({'from':'observation','value':v})                      # KEPT: always observation
            strip_c.append({'from':('human' if last_call == 'run_experiment' else 'observation'), 'value':v})  # STRIPPED: test=user
    if not have_user:
        continue

    kc = end_on_assistant([dict(c) for c in kept_c])
    if len(kc) >= 2:
        kept.append({'conversations':kc, 'system':sysp, 'tools':tools_str, '_kind':'agentic_kept','_id':task})
        stats['agentic_kept'] += 1
    stats['agentic_stripped'] += emit_prefixes(strip_c, sysp, stripped, 'agentic_stripped', task, tools=tools_str)

# ---------- write ----------
os.makedirs('sft', exist_ok=True)
def dump(rows, path):
    with open(path, 'w', encoding='utf-8') as f:
        for ex in rows:
            f.write(json.dumps({k:v for k,v in ex.items() if not k.startswith('_')}, ensure_ascii=False) + "\n")
dump(kept, 'sft/innovation_sft_kept.jsonl')
dump(stripped, 'sft/innovation_sft_stripped.jsonl')
print(f"kept     (mask_history=False): {len(kept)} examples -> sft/innovation_sft_kept.jsonl")
print(f"stripped (mask_history=True) : {len(stripped)} examples -> sft/innovation_sft_stripped.jsonl")
for k, v in stats.items():
    print(f"  {k}: {v}")
json.dump({'kept':kept,'stripped':stripped}, open('/tmp/sr_build/_sft_examples.json','w'), ensure_ascii=False)
