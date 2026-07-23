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

REPO = os.environ.get('INNOVATION_PRIOR_REPO') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

# ---------- decontamination gate (see experiments/DATA_LEAKAGE_AUDIT_zh.md) ----------
# Build-time removal of eval-benchmark leakage. Rules live in decontam/decontam_rules.json
# (regenerate the audit with decontam/audit_leakage.py). Policy (2026-07-08 user directive):
#   drop_method_slugs : discovery heuristic-search / record constructions + AHC039 -> not emitted.
#   drop_traj_slugs   : discovery-math ladders + AHC039 trajectory -> not emitted.
#   type1_finale_traj : MLS same-task trajectories that inject a NON-native stronger baseline as a
#                       finale rung -> KEEP the trajectory but SKIP the finale rung only.
# Everything else (MLS baseline ladders, standalone paper methods, FCS-research paper reconstructions,
# v4/wave2 synthetic) is kept. Toggle off with INNOVATION_DECONTAM=0 to get the pre-audit build.
DECON = os.environ.get('INNOVATION_DECONTAM', '1') != '0'
_DROP_METHODS, _DROP_TRAJ, _TYPE1_FINALE = set(), set(), set()
if DECON:
    _rp = os.path.join(REPO, 'decontam', 'decontam_rules.json')
    if os.path.isfile(_rp):
        _rules = json.load(open(_rp))
        _DROP_METHODS = set(_rules.get('drop_method_slugs', []))
        _DROP_TRAJ = set(_rules.get('drop_traj_slugs', []))
        _TYPE1_FINALE = set(_rules.get('type1_finale_traj', []))
    else:
        print(f"[decontam] WARNING: {_rp} missing; building WITHOUT the leakage gate.")
        DECON = False
_decon_stats = {'methods_dropped': 0, 'trajs_dropped': 0, 'finale_rungs_dropped': 0}

# NOTE (data remediation, see experiments/DATA_REMEDIATION_zh.md §1.3 / §3-A1):
# the old system prompt was a pure "research-register" amplifier ("You are a good researcher.")
# with no delivery discipline. It is augmented below to also carry the missing skill the FCS/ALE
# regression traces to: ship a single self-contained runnable solution that respects the I/O
# contract, and fall back to the simplest correct approach when an idea is not converging. This is
# an *instruction* consistent with the existing targets (which do end on runnable code), not a
# description of them, so it does not widen the SFT off-policy gap.
_DELIVERY = (" When you write code, deliver a single, self-contained, runnable implementation that "
             "respects any stated input/output contract; if an idea is not converging within the "
             "budget, fall back to the simplest correct approach and ship that.")
METHOD_SYS = "It is now year {year}. You are a good researcher." + _DELIVERY
TRAJ_SYS = "It is now year {year}. You are a good researcher." + _DELIVERY
AGENT_SYS = ("It is now year {year}. You are a good researcher. You work by editing the model "
            "code and running experiments. Make one tool call at a time. After editing, run an "
            "experiment to measure your method." + _DELIVERY)

# Output-format conditioning appended to the INPUT (human turn) of method/trajectory examples.
# The target answer explains the analysis in a flowing first-person voice and then ends on the
# concrete deliverable. The OLD note said "narrative, telling tone rather than a heavily formatted
# writeup" -- that phrasing (flagged in CASE_STUDY §9.5 as a smoking-gun amplifier) trains the model
# to *talk* rather than *deliver*. The note below KEEPS the first-person/flowing register (the source
# of the MLS gains) but reframes the landing as a deliverable: finish on complete, self-contained,
# runnable code, not a sketch. Conditional because some examples have no code scaffold.
_NOTE_CODE = ("\n\nFill in the scaffold above into a complete, self-contained, runnable "
              "implementation. Explain the analysis in a flowing, first-person tone, then finish on "
              "the final working code as your deliverable -- not a sketch or a fragment.")
_NOTE_NOCODE = ("\n\nExplain the analysis in a flowing, first-person tone, then land on a clean, "
                "precise final statement of the result.")
def fmt_note(input_text):
    """Pick the format note: scaffold-fill variant iff the input contains a code block."""
    return _NOTE_CODE if '```' in (input_text or '') else _NOTE_NOCODE
def read_with_note(p):
    """read() the input file and append the output-format conditioning note."""
    t = read(p)
    return t + fmt_note(t)

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

_THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
def fold_think(value):
    """Remove the <think>...</think> block ENTIRELY from a folded HISTORY turn (keep the answer/action).

    Previously this left an EMPTY `<think>\\n\\n</think>` in place. Training conditioned the model on
    thousands of those empty-think history blocks (2738 in the built file), and methodtraj/agentic
    models learned to NARRATE the empty think and bail mid-trace -- e.g. "I cannot complete this thought
    because the next thinking is empty </think>" then dump code, observed in 159/2955 real eval samples
    (concentrated in the methodtraj models that carry the most folded turns; see
    experiments/DATA_REMEDIATION_zh.md). Dropping the block removes that conditioning signal while still
    giving the current turn a reasoning-stripped history (the whole intent of folding). The current
    (loss=True) turn is unchanged and keeps its full reasoning."""
    return _THINK_RE.sub('', value, count=1).lstrip('\n')
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
# train_answer.md code-integrity gate. The discovery-writeup contract says train_answer's code is
# copied VERBATIM from answer.md (the reviewed canonical deliverable); the 2026-07 audit found 548
# methods violating it — code invented from thin air (theory methods whose answer has no code) or
# silently re-implemented (never executed, never reviewed; sampled files had hard bugs). For those
# suspect slugs the answer channel uses train_answer ONLY IF every large code fence in it is a
# whitespace-normalized substring of answer.md's fences; otherwise it presents answer.md itself.
# Self-healing: a train_answer later fixed to verbatim (or to a codeless artifact ending where
# answer.md has no code) is used again automatically. Missing train_answer falls back to answer.md.
_sus_f = 'sft/train_answer_suspects.txt'
_TA_SUSPECTS = ({x.strip() for x in open(_sus_f) if x.strip() and not x.startswith('#')}
                if os.path.isfile(_sus_f) else set())
_FENCE_RE = re.compile(r'```[a-zA-Z0-9_+-]*\n(.*?)```', re.S)
def _ta_code_ok(ta_text, ans_text):
    hay = re.sub(r'\s+', '', ''.join(_FENCE_RE.findall(ans_text)))
    for b in _FENCE_RE.findall(ta_text):
        nb = re.sub(r'\s+', '', b)
        if len(nb) >= 200 and nb not in hay:
            return False
    return True
methods = json.load(open('methods.json'))
for m in methods:
    slug, yr = m['slug'], m.get('year')
    if slug in _DROP_METHODS:                     # decontam: eval-task reconstruction
        _decon_stats['methods_dropped'] += 1
        continue
    d = f'methods/{slug}/results'
    if not all(os.path.isfile(f'{d}/{f}.md') for f in ('context', 'reasoning')):
        continue
    has_ta, has_ans = os.path.isfile(f'{d}/train_answer.md'), os.path.isfile(f'{d}/answer.md')
    if not has_ta and not has_ans:
        continue
    if not has_ta:
        ans_text = read(f'{d}/answer.md')
        stats['method_answer_fallback'] = stats.get('method_answer_fallback', 0) + 1
    else:
        ans_text = read(f'{d}/train_answer.md')
        if slug in _TA_SUSPECTS and has_ans and not _ta_code_ok(ans_text, read(f'{d}/answer.md')):
            ans_text = read(f'{d}/answer.md')
            stats['method_ta_bypass'] = stats.get('method_ta_bypass', 0) + 1
    examples.append({'conversations': [{'from':'human','value':read_with_note(f'{d}/context.md')},
                                        {'from':'gpt','value':think(read(f'{d}/reasoning.md'), ans_text)}],
                     'system': METHOD_SYS.format(year=yr), '_kind':'method', '_id':slug})
    stats['method'] += 1

# ---------- (a2) v4 competition deliverables : single-turn, OVERSAMPLED into the mix ----------
# The data_v4/ traces (single-file C++ reading stdin, debug+self-verify spine, no class) were NOT in
# the SFT mix at all -- build_sft never read them and build_v4.py wrote a separate, untrained file
# (experiments/DATA_FIX_FCS_LANDING_zh.md P0.2 flags this: "v4 is 0%, not 13%"). They are the cleanest
# "executable deliverable" signal, so we ingest the VERIFIED ones here -- ONCE each, NO row duplication.
# (To up-weight v4, use a dataset-level sampling weight / interleave_probs at train time, never copy rows.)
V4_SYS = ("It is now year {year}. You are an expert competitive programmer. Solve the problem with a "
          "single, self-contained C++ program that reads from standard input and writes to standard "
          "output. Before committing, verify your reasoning and your code: trace it on concrete inputs, "
          "check the edge cases, and fix any bug you find. Output the final solution as one C++ block.")
_v4_verified = {'cp-noadj-commit'}  # the hand-authored flagship + the locally oracle-verified lists
for _vf in ('data_v4/_verified.txt', 'data_v4/_cpv4b_verified.txt'):
    if os.path.isfile(_vf):
        _v4_verified |= {x.strip() for x in open(_vf) if x.strip()}
# the 100 new innovation-emphasized FrontierCS (fcs-*) + ALE (ale-*) datapoints from
# PROBLEM_CANDIDATES.md: each was self-verified (compile+oracle / scorer) AND independently
# Codex-reviewed at generation time (tools/gen_problems_workflow.js, gen_ale_workflow.js).
for _g in glob.glob('data_v4/fcs-*/') + glob.glob('data_v4/ale-*/'):
    _v4_verified.add(_g.rstrip('/').split('/')[-1])
v4_n = 0
for slug in sorted(_v4_verified):
    d = f'data_v4/{slug}'
    if not all(os.path.isfile(f'{d}/{f}.md') for f in ('context', 'reasoning', 'train_answer')):
        continue
    examples.append({'conversations': [{'from':'human','value':read_with_note(f'{d}/context.md')},
                                        {'from':'gpt','value':think(read(f'{d}/reasoning.md'), read(f'{d}/train_answer.md'))}],
                     'system': V4_SYS.format(year=2025), '_kind':'v4', '_id':slug})
    v4_n += 1
stats['v4_unique'] = v4_n

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
    if task in _DROP_TRAJ:                         # decontam: discovery ladder / AHC039 reconstruction
        _decon_stats['trajs_dropped'] += 1
        continue
    tj = trajs.get(task); yr = tj['year'] if tj else None
    d = f'trajectories/{task}'
    meta = json.load(open(meta_p))
    steps = [s for s in sorted(meta.get('steps', []), key=lambda s: s.get('n', 0)) if s.get('reasoning')]
    if task in _TYPE1_FINALE:                      # decontam: drop the injected non-native finale rung
        _n0 = len(steps)
        steps = [s for s in steps if not s.get('finale')]
        _decon_stats['finale_rungs_dropped'] += _n0 - len(steps)
    if not steps:
        continue
    init = read_with_note(f"{d}/{meta.get('initial_context_file','00-initial-context.md')}")
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
    if task in _DROP_TRAJ:                         # decontam: keep in sync with trajectory gate
        continue
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

# ---------- per-example tags (for mix control; see DATA_REMEDIATION_zh.md §3-A1/A3) ----------
# Tag each example by what its TRAINED landing looks like, so the SFT mix can cap research-narrative
# and require a floor of executable-deliverable samples. We inspect only the gpt/function_call turns.
_STDIN_RE = re.compile(r'\b(std::cin|cin\s*>>|scanf|getline|sys\.stdin|input\(\)|readline\(\))')
_CPP_RE = re.compile(r'```(?:cpp|c\+\+|cc)\b', re.I)
_CLASS_RE = re.compile(r'```[^\n]*\n(?:.*\n)*?\s*class\s+\w+', re.I)
_FALLBACK_RE = re.compile(r'(fall back|fallback|simplest correct|keep it simple|revert to|stick with '
                          r'the simple|too risky|not worth the risk|ship the simple|retreat to)', re.I)
def tag_example(ex):
    txt = "\n".join(t['value'] for t in ex['conversations'] if t['from'] in ('gpt', 'function_call'))
    return {'id': ex.get('_id'), 'kind': ex.get('_kind'),
            'reads_stdin': bool(_STDIN_RE.search(txt)), 'has_cpp': bool(_CPP_RE.search(txt)),
            'defines_class': bool(_CLASS_RE.search(txt)), 'has_fallback': bool(_FALLBACK_RE.search(txt)),
            'has_code': '```' in txt}

# ---------- uniform schema (load-safety, 2026-07) ----------
# A file that uses the per-turn `loss` flag ANYWHERE must carry it EVERYWHERE, and every optional
# top-level column (`tools`) must exist on every row. HF datasets/pyarrow infer the arrow schema
# from the FIRST block and cast the rest: with mixed rows the absent `loss` key materializes as
# `loss: None` on every unflagged turn -- and the fork's old `bool(message.get("loss", True))`
# turned None into False, silently training those examples with ZERO loss -- or the load crashes
# outright on the schema cast (both observed with datasets 4.8.5 on this exact file; which one you
# get depends on row order). The fork treats None as "train" since commit 494ff82; the explicit
# fields below make the load independent of datasets version and row order.
for ex in examples:
    if 'tools' not in ex:
        ex['tools'] = ""
    for t in ex['conversations']:
        if t['from'] in ('gpt', 'function_call') and 'loss' not in t:
            t['loss'] = True

# ---------- HARD INVARIANT: a turn whose think was stripped by folding NEVER enters the loss ----
# fold_turn() couples the strip with loss=False; this guards against any future edit decoupling
# them. In a folded example the trained turns must be exactly the trailing current-round block,
# every folded (loss=False) turn must carry no <think>, and non-folded examples train every turn.
for ex in examples:
    ast = [t for t in ex['conversations'] if t['from'] in ('gpt', 'function_call')]
    flags = [t.get('loss', True) for t in ast]
    if str(ex.get('_kind', '')).endswith('folded'):
        assert False in flags and flags == [False] * flags.count(False) + [True] * flags.count(True), \
            f"{ex.get('_id')}: folded example must train only the trailing current round"
        assert all('<think>' not in t['value'] for t, fl in zip(ast, flags) if not fl), \
            f"{ex.get('_id')}: a loss=False folded turn still carries a <think> block"
    else:
        assert all(flags), f"{ex.get('_id')}: non-folded example must train every assistant turn"

# ---------- write ----------
# Output paths are parametrizable so a decontam build can be produced WITHOUT overwriting the
# original sft/innovation_sft.jsonl during review (set SFT_OUT / SFT_TAGS_OUT).
out = os.environ.get('SFT_OUT', 'sft/innovation_sft.jsonl')
tags_out = os.environ.get('SFT_TAGS_OUT', 'sft/_sft_tags.jsonl')
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
if DECON:
    print(f"[decontam] gate ON: dropped {_decon_stats['methods_dropped']} methods, "
          f"{_decon_stats['trajs_dropped']} trajectories, {_decon_stats['finale_rungs_dropped']} "
          f"Type-1 finale rungs. (INNOVATION_DECONTAM=0 to disable.)")
tags = []
with open(out, 'w', encoding='utf-8') as f:
    for ex in examples:
        f.write(json.dumps({k:v for k,v in ex.items() if not k.startswith('_')}, ensure_ascii=False) + "\n")
        tags.append(tag_example(ex))
with open(tags_out, 'w', encoding='utf-8') as f:
    for t in tags:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
n = len(tags) or 1
agg = {k: sum(1 for t in tags if t.get(k)) for k in ('has_code','reads_stdin','has_cpp','defines_class','has_fallback')}
print(f"wrote {out}: {len(examples)} examples (train with mask_history=False)")
print(f"wrote {tags_out} ({len(tags)} rows); landing mix:")
for k, v in agg.items():
    print(f"  {k}: {v} ({100*v/n:.1f}%)")
for k, v in stats.items():
    print(f"  {k}: {v}")
os.makedirs('/tmp/sr_build', exist_ok=True)
json.dump(examples, open('/tmp/sr_build/_sft_examples.json', 'w'), ensure_ascii=False)
