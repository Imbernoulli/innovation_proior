#!/usr/bin/env python3
"""Build the unified LLaMA-Factory ShareGPT SFT JSONL (multi-framing).

Why multi-framing (design note): in a single causal forward pass a `<think>`
block is either physically present in the sequence or not -- if present, every
later token attends to it. But the Qwen rolling-think rule wants the SAME think
visible when generating turns inside its own episode and invisible once a new
real user query opens a later episode. One sequence cannot satisfy both. So
instead of forcing one canonical representation we emit each trajectory under
SEVERAL self-consistent framings; the union trains thinking, the post-think
answer, and tool-use, and teaches the model to generalize across serving modes.

  (a) methods            -> single-turn Q&A (context -> <think>reasoning</think>train_answer).

  (b) trajectories
       view 1  "continuum"  -> full multi-turn; measured feedback is an
                               `observation` (a <tool_response>, which does NOT
                               reset the rolling checkpoint), so ALL rungs keep
                               their <think> in context and every rung is trained.
       view 2  "per-rung"   -> each rung k>=2 as a SINGLE-target sample: the prior
                               rungs (answers only, think stripped) are folded into
                               the human prompt, gpt = this rung's <think>+answer.
                               One gpt turn => correct even under mask_history=False,
                               and it trains the rung in a history-think-stripped
                               (test=user) context.

  (c) agentic            -> full multi-turn tool loop. Assistant tool steps are
                            emitted as structured `function_call` so LF renders the
                            per-model wrapper (qwen3 JSON / qwen3_5 XML) from one
                            dataset; think + say + tool-call all preserved. Tool
                            results are `observation`.

System prompt carries the discovery YEAR (method year for (a); the trajectory's
first-method year for (b)/(c)) as meta-conditioning.

Train with mask_history=False (loss on every gpt/function_call turn). Single-target
samples in view 2 stay correct because they contain exactly one gpt turn.
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

# Replace literal structural/template tokens inside EMBEDDED content so they can't
# collide with the wrappers we add (e.g. a method whose text discusses <think>).
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

examples = []
stats = {'method':0,'traj_continuum':0,'traj_perrung':0,'agentic':0,
         'method_turns':0,'traj_turns':0,'perrung_turns':0,'agent_turns':0,'agent_calls':0}

# ---------- (a) methods : single-turn ----------
methods = json.load(open('methods.json'))
for m in methods:
    slug, yr = m['slug'], m.get('year')
    d = f'methods/{slug}/results'
    if not all(os.path.isfile(f'{d}/{f}.md') for f in ('context','reasoning','train_answer')):
        continue
    examples.append({
        'conversations': [{'from':'human','value':read(f'{d}/context.md')},
                          {'from':'gpt','value':think(read(f'{d}/reasoning.md'), read(f'{d}/train_answer.md'))}],
        'system': METHOD_SYS.format(year=yr), '_kind':'method', '_id':slug})
    stats['method'] += 1; stats['method_turns'] += 2

# ---------- (b) trajectories ----------
trajs = {x['task']: x for x in json.load(open('trajectories.json'))}
def step_answer(d, st):
    """train_answer for the rung if present, else the markdown answer."""
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

    # view 1 -- continuum: feedback as observation (keeps every rung's think)
    convs = [{'from':'human','value':init}]
    for st in steps:
        convs.append({'from':'gpt','value':think(read(f"{d}/{st['reasoning']}"), step_answer(d, st))})
        fb = st.get('feedback')
        if fb and os.path.isfile(f"{d}/{fb}"):
            convs.append({'from':'observation','value':read(f"{d}/{fb}")})
    while convs and convs[-1]['from'] != 'gpt':   # must end on gpt
        convs.pop()
    if len(convs) >= 2:
        examples.append({'conversations':convs, 'system':TRAJ_SYS.format(year=yr),
                         '_kind':'traj_continuum','_id':task})
        stats['traj_continuum'] += 1; stats['traj_turns'] += len(convs)

    # view 2 -- per-rung: each rung k>=2 in a fresh, history-think-stripped context
    for k in range(1, len(steps)):           # k indexes the TARGET rung (0-based); start at rung 2
        recap = [init]
        for j in range(k):
            sj = steps[j]
            method = sj.get('method') or sj.get('slug') or f'round {j+1}'
            block = f"## Round {j+1}: {neutralize(str(method))}\n\n{step_answer(d, sj)}"
            fb = sj.get('feedback')
            if fb and os.path.isfile(f"{d}/{fb}"):
                block += f"\n\n### Measured result\n\n{read(f'{d}/{fb}')}"
            recap.append(block)
        human = ("\n\n---\n\n".join(recap)
                 + "\n\n---\n\nGiven the rounds so far and their measured results, "
                   "reason about what to change and propose the next improvement.")
        gpt = think(read(f"{d}/{steps[k]['reasoning']}"), step_answer(d, steps[k]))
        examples.append({'conversations':[{'from':'human','value':human},{'from':'gpt','value':gpt}],
                         'system':TRAJ_SYS.format(year=yr), '_kind':'traj_perrung','_id':f'{task}#r{k+1}'})
        stats['traj_perrung'] += 1; stats['perrung_turns'] += 2

# ---------- (c) agentic : structured function_call ----------
def fc_value(msg):
    parts = []
    rc = (msg.get('reasoning_content') or '').strip()
    if rc:
        parts.append(f"<think>\n{neutralize(rc)}\n</think>")
    ct = (msg.get('content') or '').strip()
    if ct:
        parts.append(neutralize(ct))
    fn = msg['tool_calls'][0]['function']           # verified: no parallel calls
    call = json.dumps({'name': fn['name'], 'arguments': fn['arguments']}, ensure_ascii=False)
    parts.append(f"<tool_call>\n{call}\n</tool_call>")
    return "\n\n".join(parts)

def gpt_value(msg):
    rc = (msg.get('reasoning_content') or '').strip()
    ct = (msg.get('content') or '').strip()
    return f"<think>\n{neutralize(rc)}\n</think>\n\n{neutralize(ct)}" if rc else neutralize(ct)

for ap in sorted(glob.glob('trajectories/*/agentic_messages.json')):
    task = os.path.basename(os.path.dirname(ap))
    yr = trajs[task]['year'] if task in trajs else None
    data = json.load(open(ap))
    tools_str = json.dumps(data.get('tools', []), ensure_ascii=False)   # full {type,function} objs
    convs = []
    for msg in data['messages']:
        role = msg['role']
        if role == 'system':
            continue
        if role == 'user':
            convs.append({'from':'human','value':neutralize(msg['content'])})
        elif role == 'tool':
            convs.append({'from':'observation','value':neutralize(msg['content'])})
        elif role == 'assistant':
            if msg.get('tool_calls'):
                convs.append({'from':'function_call','value':fc_value(msg)}); stats['agent_calls'] += 1
            else:
                v = gpt_value(msg)
                if v.strip():
                    convs.append({'from':'gpt','value':v})
    while convs and convs[-1]['from'] in ('human','observation'):   # end on assistant-side
        convs.pop()
    if len(convs) < 2:
        continue
    examples.append({'conversations':convs, 'system':AGENT_SYS.format(year=yr),
                     'tools':tools_str, '_kind':'agentic','_id':task})
    stats['agentic'] += 1; stats['agent_turns'] += len(convs)

# ---------- write ----------
os.makedirs('sft', exist_ok=True)
out = 'sft/innovation_sft.jsonl'
with open(out, 'w', encoding='utf-8') as f:
    for ex in examples:
        rec = {k: v for k, v in ex.items() if not k.startswith('_')}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"wrote {out}: {len(examples)} examples")
for k, v in stats.items():
    print(f"  {k}: {v}")
json.dump(examples, open('/tmp/sr_build/_sft_examples.json', 'w'), ensure_ascii=False)
