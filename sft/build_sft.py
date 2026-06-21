#!/usr/bin/env python3
"""Build ONE LLaMA-Factory ShareGPT SFT file with reasoning folding baked into the data.

ONE file, ONE training config. Real multi-turn roles throughout. The two input distributions
the model must learn are produced by emitting each ladder/trace under two framings, using the
per-turn `loss` flag (the LlamaFactory `feat/per-turn-loss-mask` fork -- see sft/README.md):

  Mode 1  "full"   : the whole conversation, every turn keeps its real <think> and is trained
                     (no loss flags).  => history WITH reasoning.
  Mode 2  "folded" : for each round taken as the CURRENT round, the prior rounds keep their
                     answers / actions / results but their <think> content is EMPTIED to
                     `<think>\\n\\n</think>` AND marked `loss=False` (kept as context, NOT
                     trained); the current round keeps ALL its reasoning (no intra-round folding
                     -- the long first-turn reasoning stays) and is marked `loss=True` (every one
                     of its actions is trained).  => current round derived against a
                     reasoning-stripped history.

A "round" = one rung (trajectory) or one run_experiment-delimited block (agentic). The same
folding logic applies to both. The per-turn `loss` flag is what guarantees "all of the current
round's actions train, the folded history does not" -- mask_history can't express that (it is
all-turns or last-turn-only). Train with mask_history=False (default); the loss flags do the work.

  (a) methods   : single-turn Q&A (full reasoning).
  (b) trajectory: Mode 1 (feedback = observation) + Mode 2 (prior rungs folded, feedback = user
                  boundary, current rung full).
  (c) agentic   : Mode 1 (all results = observation; structured function_call -> qwen3 JSON /
                  qwen3_5 XML) + Mode 2 (prior rounds folded with run_experiment result = user
                  boundary and str_replace result = observation; current round full, its closing
                  run_experiment result dropped as the next boundary).

System prompt carries the discovery YEAR (method year; trajectory first-method year).
"""
import json, os, glob, re

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

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
def fold_think(value):
    """Empty the <think> content (keep the tags + everything after) -- the non-thinking form."""
    return _THINK_RE.sub('<think>\n\n</think>', value, count=1)
def fold_turn(turn):
    t = dict(turn)
    if t['from'] in ('gpt', 'function_call'):
        t['value'] = fold_think(t['value'])
        t['loss'] = False          # folded history: kept as context, NOT trained
    return t

examples = []
stats = {'method':0,'traj_full':0,'traj_folded':0,'agentic_full':0,'agentic_folded':0}

def end_on_assistant(convs):
    while convs and convs[-1]['from'] in ('human', 'observation'):
        convs.pop()
    return convs

# ---------- (a) methods : single-turn ----------
methods = json.load(open('methods.json'))
for m in methods:
    slug, yr = m['slug'], m.get('year')
    d = f'methods/{slug}/results'
    if not all(os.path.isfile(f'{d}/{f}.md') for f in ('context', 'reasoning', 'train_answer')):
        continue
    examples.append({'conversations': [{'from':'human','value':read(f'{d}/context.md')},
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
    rungs = []  # (gpt_turn, feedback_text|None)
    for st in steps:
        gpt = {'from':'gpt','value':think(read(f"{d}/{st['reasoning']}"), step_answer(d, st))}
        fb = st.get('feedback')
        fbtext = read(f"{d}/{fb}") if (fb and os.path.isfile(f"{d}/{fb}")) else None
        rungs.append((gpt, fbtext))

    # Mode 1 (full): feedback = observation
    convs = [{'from':'human','value':init}]
    for gpt, fbtext in rungs:
        convs.append(dict(gpt))
        if fbtext is not None:
            convs.append({'from':'observation','value':fbtext})
    convs = end_on_assistant(convs)
    if len(convs) >= 2:
        examples.append({'conversations':convs, 'system':sysp, '_kind':'traj_full','_id':task})
        stats['traj_full'] += 1

    # Mode 2 (folded): prior rungs folded (empty think) with feedback = user boundary; current rung full
    for c in range(1, len(rungs)):
        convs = [{'from':'human','value':init}]
        for j in range(c):                                  # prior rungs -> folded, feedback = user
            convs.append(fold_turn(rungs[j][0]))
            if rungs[j][1] is not None:
                convs.append({'from':'human','value':rungs[j][1]})
        cur = dict(rungs[c][0]); cur['loss'] = True         # current rung -> full reasoning, trained
        convs.append(cur)
        examples.append({'conversations':convs, 'system':sysp, '_kind':'traj_folded','_id':f'{task}#r{c+1}'})
        stats['traj_folded'] += 1

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

def parse_rounds(msgs):
    init = None; rounds = []; cur = []
    for m in msgs:
        r = m['role']
        if r == 'system':
            continue
        if r == 'user':
            init = m['content']
        elif r == 'assistant':
            calls = m.get('tool_calls', [])
            fn = calls[0]['function'] if calls else None
            cur.append({'msg': m, 'call': fn['name'] if fn else None})
        elif r == 'tool':
            if cur:
                cur[-1]['result'] = m['content']
                if cur[-1]['call'] == 'run_experiment':
                    rounds.append(cur); cur = []
    if cur:
        rounds.append(cur)
    return init, rounds

def agentic_round(actions, fold, is_current):
    """fold=True -> empty think (history round). is_current -> drop the closing run_experiment
    result (it is the boundary to the next round). History run_experiment result = `user`
    boundary; str_replace result = observation."""
    out = []
    for a in actions:
        turn = {'from':'function_call','value':fc_value(a['msg'])}
        if fold:
            turn = fold_turn(turn)                 # folded history -> loss=False
        elif is_current:
            turn['loss'] = True                    # current round -> all actions trained
        out.append(turn)
        if 'result' in a:
            if a['call'] == 'run_experiment':
                if not is_current:
                    out.append({'from':'human','value':neutralize(a['result'])})   # boundary
            else:
                out.append({'from':'observation','value':neutralize(a['result'])})
    return out

for ap in sorted(glob.glob('trajectories/*/agentic_messages.json')):
    task = os.path.basename(os.path.dirname(ap))
    yr = trajs[task]['year'] if task in trajs else None
    data = json.load(open(ap))
    tools_str = json.dumps(data.get('tools', []), ensure_ascii=False)
    sysp = AGENT_SYS.format(year=yr)
    init, rounds = parse_rounds(data['messages'])
    if init is None or not rounds:
        continue

    # Mode 1 (full): all results = observation, full reasoning
    convs = [{'from':'human','value':neutralize(init)}]
    for actions in rounds:
        for a in actions:
            convs.append({'from':'function_call','value':fc_value(a['msg'])})
            if 'result' in a:
                convs.append({'from':'observation','value':neutralize(a['result'])})
    convs = end_on_assistant(convs)
    if len(convs) >= 2:
        examples.append({'conversations':convs, 'system':sysp, 'tools':tools_str,
                         '_kind':'agentic_full','_id':task})
        stats['agentic_full'] += 1

    # Mode 2 (folded): prior rounds folded (boundary = user); current round full, closing result dropped
    for c in range(1, len(rounds)):
        convs = [{'from':'human','value':neutralize(init)}]
        for j in range(c):
            convs += agentic_round(rounds[j], fold=True, is_current=False)
        convs += agentic_round(rounds[c], fold=False, is_current=True)
        convs = end_on_assistant(convs)
        if len(convs) >= 2:
            examples.append({'conversations':convs, 'system':sysp, 'tools':tools_str,
                             '_kind':'agentic_folded','_id':f'{task}#r{c+1}'})
            stats['agentic_folded'] += 1

# ---------- write ----------
os.makedirs('sft', exist_ok=True)
out = 'sft/innovation_sft.jsonl'
with open(out, 'w', encoding='utf-8') as f:
    for ex in examples:
        f.write(json.dumps({k:v for k,v in ex.items() if not k.startswith('_')}, ensure_ascii=False) + "\n")
print(f"wrote {out}: {len(examples)} examples (train with mask_history=False)")
for k, v in stats.items():
    print(f"  {k}: {v}")
json.dump(examples, open('/tmp/sr_build/_sft_examples.json', 'w'), ensure_ascii=False)
