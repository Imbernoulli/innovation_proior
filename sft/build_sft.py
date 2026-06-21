#!/usr/bin/env python3
"""Build the unified LLaMA-Factory ShareGPT SFT JSONL (multi-framing).

Grounded in the Qwen3 / Qwen3.5 chat templates + model cards:
  * One hybrid model. Thinking is toggled per-request by `enable_thinking`
    (closed empty <think></think> = off; open <think> = on). Qwen3.5 dropped the
    /think /no_think soft tokens; the history-stripping behaviour is identical to Qwen3.
  * History stripping is ALWAYS on and independent of the toggle: the template removes
    <think> from assistant turns at/before the most recent REAL user query
    (a `<tool_response>` / observation does NOT count as a user query, so reasoning is
    RETAINED inside a tool loop and only wiped when a new user message arrives).
  * Official guidance: history "should only include the final output part and does not
    need to include the thinking content" -- but frameworks that don't run the Jinja must
    enforce this themselves.

So the model must learn BOTH input distributions, and we emit each source under the
framing whose train-render == its inference-render:

  (a) methods            -> single-turn Q&A (context -> <think>reasoning</think>train_answer).

  (b) trajectories  (uniform feedback; never mixes user+observation)
       continuum  -> full multi-turn; feedback = `observation` (keeps every rung's
                     <think>; matches a tool-loop serving where results come back as
                     observations).  [WITH history reasoning]
       per-rung   -> each rung k>=2 as a single-target sample; prior rungs (answers,
                     think stripped) folded into the human prompt; gpt = this rung's
                     <think>+answer.  [WITHOUT history reasoning -- the post-user-query
                     default]

  (c) agentic  (the genuinely MIXED case: str_replace results are observations,
                run_experiment results are the test feedback / round boundaries)
       continuum  -> full tool loop; ALL results (incl. run_experiment) = `observation`,
                     structured `function_call` steps (LF renders qwen3 JSON / qwen3_5 XML
                     from one file). Reasoning retained throughout = a real never-reset
                     agent. [WITH history reasoning]
       per-round  -> run_experiment result treated as a USER boundary; str_replace results
                     stay `observation`. Prior rounds fold into the human prompt (stripped);
                     the current round's edit-loop + test call are real turns, all trained.
                     [WITHOUT history reasoning across test boundaries]

Folding each sample's pre-boundary history into the human opening makes EVERY sample a
single rolling-checkpoint episode (no internal user turn), so `mask_history=False`
(loss on every gpt/function_call turn) is correct and uniform for all kinds.

System prompt carries the discovery YEAR (method year for (a); trajectory first-method
year for (b)/(c)) as meta-conditioning.
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
stats = {'method':0,'traj_continuum':0,'traj_perrung':0,'agentic_continuum':0,'agentic_perround':0,
         'method_turns':0,'traj_turns':0,'perrung_turns':0,'agent_turns':0,'agent_calls':0,'perround_turns':0}

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

    # continuum: feedback as observation (keeps every rung's think)
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

    # per-rung: each rung k>=2 in a fresh, history-think-stripped context
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

# ---------- (c) agentic : structured function_call, two framings ----------
def fc_value(msg):
    """assistant tool step -> <think>..</think>{say}<tool_call>{json}</tool_call> (one call/turn)."""
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

def parse_rounds(msgs):
    """Linearise into rounds. A round = the actions up to and INCLUDING a run_experiment
    call; the run_experiment result (metrics) closes it and opens the next round."""
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
            args = fn['arguments'] if fn else {}
            cur.append({'msg': m,
                        'call': fn['name'] if fn else None,
                        'path': args.get('path') if isinstance(args, dict) else None})
        elif r == 'tool':
            if cur:
                cur[-1]['result'] = m['content']
                if cur[-1]['call'] == 'run_experiment':
                    rounds.append({'actions': cur, 'metrics': m['content']}); cur = []
    if cur:
        rounds.append({'actions': cur, 'metrics': None})
    return init, rounds

def round_turns(rd, drop_final_run_result):
    """Render a round's actions as conv turns. str_replace results -> observation;
    a closing run_experiment result is dropped when it is the boundary to the next round."""
    out = []
    for a in rd['actions']:
        out.append({'from': 'function_call', 'value': fc_value(a['msg'])})
        if 'result' in a and not (drop_final_run_result and a['call'] == 'run_experiment'):
            out.append({'from': 'observation', 'value': neutralize(a['result'])})
    return out

def summarize_round(rd, idx):
    """Compact, think-stripped recap of a prior round for folding into a human prompt."""
    lines = [f"## Round {idx}"]
    for a in rd['actions']:
        say = (a['msg'].get('content') or '').strip()
        if say:
            lines.append(neutralize(say))
        if a['call'] == 'str_replace':
            lines.append(f"→ edited `{a.get('path') or '?'}`")
        elif a['call'] == 'run_experiment':
            lines.append("→ ran experiment")
        else:
            lines.append(f"→ {a['call']}")
    if rd.get('metrics'):
        lines.append("### Measured result\n\n" + neutralize(rd['metrics']))
    return "\n\n".join(lines)

for ap in sorted(glob.glob('trajectories/*/agentic_messages.json')):
    task = os.path.basename(os.path.dirname(ap))
    yr = trajs[task]['year'] if task in trajs else None
    data = json.load(open(ap))
    tools_str = json.dumps(data.get('tools', []), ensure_ascii=False)   # full {type,function} objs
    init, rounds = parse_rounds(data['messages'])
    if init is None or not rounds:
        continue

    # continuum: ALL results (incl. run_experiment) = observation; reasoning retained throughout
    convs = [{'from': 'human', 'value': neutralize(init)}]
    for rd in rounds:
        convs += round_turns(rd, drop_final_run_result=False)
    while convs and convs[-1]['from'] in ('human', 'observation'):
        convs.pop()
    if len(convs) >= 2:
        examples.append({'conversations': convs, 'system': AGENT_SYS.format(year=yr),
                         'tools': tools_str, '_kind': 'agentic_continuum', '_id': task})
        stats['agentic_continuum'] += 1; stats['agent_turns'] += len(convs)
        stats['agent_calls'] += sum(1 for c in convs if c['from'] == 'function_call')

    # per-round (test=user): each round i>=2 with prior rounds folded into the human prompt
    for i in range(1, len(rounds)):
        recap = [neutralize(init)] + [summarize_round(rounds[j], j + 1) for j in range(i)]
        human = ("\n\n---\n\n".join(recap)
                 + "\n\n---\n\nGiven the edits so far and the measured results, reason about what to "
                   "change next, make the edits, then run the experiment.")
        convs = [{'from': 'human', 'value': human}]
        convs += round_turns(rounds[i], drop_final_run_result=True)
        while convs and convs[-1]['from'] in ('human', 'observation'):
            convs.pop()
        if len(convs) >= 2:
            examples.append({'conversations': convs, 'system': AGENT_SYS.format(year=yr),
                             'tools': tools_str, '_kind': 'agentic_perround', '_id': f'{task}#r{i+1}'})
            stats['agentic_perround'] += 1; stats['perround_turns'] += len(convs)

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
