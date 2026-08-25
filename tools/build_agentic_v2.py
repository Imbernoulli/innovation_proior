#!/usr/bin/env python3
"""Rebuild the agentic transcripts against the CURRENT MLS-Bench devfix harness contract.

Replaces the 2026-06-21 agentic_messages.json generation (tools str_replace/create/
run_experiment, one giant think on the first edit, 428 zero-think boilerplate turns)
with per-task `agentic_v2.json` skeletons that

  * re-derive every code state from the CURRENT trajectory files (rung answer.md
    largest code fence; scaffold from 00-initial-context.md), so the 2026-07/08
    reasoning/answer updates are inherited for free;
  * speak the devfix eval contract (2026-08-23 user ruling, .cache/mlsbench-eval):
    edit(op='str_replace'|'create', filename, ...) + view + test()/#N + submit(n) +
    undo, tolerant-matcher system prompt, initial prompt with '%6d: ' line-number
    file dump and the real "## Your Budget" block;
  * decompose each rung transition into unique-match str_replace ops via difflib,
    replayed after every op — the build FAILS unless the final replayed state is
    byte-identical to the target rung code (same guarantee the June build had);
  * leave think/say SLOTS (`[[THINK kind=... rung=N slug=...]]`) on every assistant
    turn for the prose workflow to fill. NO turn is allowed to stay empty: the
    downstream lint (--lint) rejects any remaining sentinel and any empty
    reasoning_content, which is what kills the 33.6%-zero-think / empty-<think>
    supervision problem of the June data.

Scope: every trajectories/*/meta.json ladder EXCEPT decontam drop_traj_slugs;
type1_finale tasks drop the injected finale rung (same policy as build_sft.py).
Tasks whose states can't be extracted or verified are reported, never half-built.

Usage:
  python3 tools/build_agentic_v2.py            # build all, write agentic_v2.json + report
  python3 tools/build_agentic_v2.py TASK ...   # build selected tasks
  python3 tools/build_agentic_v2.py --lint     # verify filled files (no sentinel, no
                                               # empty think, ops still replay clean)
"""
import difflib
import glob
import json
import os
import re
import sys

REPO = os.environ.get('INNOVATION_PRIOR_REPO') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

FENCE = re.compile(r'```([a-zA-Z0-9_+-]*)\n(.*?)```', re.S)
OUT_NAME = 'agentic_v2.json'
REPORT = 'tools/agentic_v2_report.json'

# ---------------------------------------------------------------------------
# decontam (mirror build_sft.py)
# ---------------------------------------------------------------------------
_rules = json.load(open('decontam/decontam_rules.json'))
DROP_TRAJ = set(_rules.get('drop_traj_slugs', []))
TYPE1_FINALE = set(_rules.get('type1_finale_traj', []))

# ---------------------------------------------------------------------------
# Harness contract literals — the LIVE three-way protocol (eval == RL == SFT).
# Source of truth: .cache/mlsbench-eval (local commit 2861229a4) DEFAULT state:
# MLSBENCH_STRICT_STR_REPLACE unset, MLSBENCH_VIEW_TOOL=1, MLSBENCH_REWRITE_OP=1
# — i.e. the Cline-style op split (rewrite = the normal way, str_replace = 1-10
# line surgical fixes, create) + view, tolerant matcher, and the post-edit echo
# of the file's current editable region. The RL episode worker instantiates the
# same InteractiveAgent, so these strings are what BOTH training-time rollouts
# and eval-time episodes actually see. System prompt = SYSTEM_PROMPT_SCI with
# the rewrite-branch replace_block (interactive.py); tool schemas verbatim from
# tools.py (EDIT_REWRITE_SCHEMA / VIEW_SCHEMA / TOOL_SCHEMAS); result strings
# and the echo format verbatim from tools.py edit()/_file_snapshot().
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an ML scientist. Your goal is to propose and implement a novel algorithmic \
contribution that improves performance on the given task.

What counts as a good contribution:
- A new loss function or objective formulation
- A new policy update rule or gradient estimation method
- A novel exploration or regularization strategy
- A new way to parameterize or combine components, with clear motivation

What does NOT count:
- Trivially increasing network capacity to brute-force a metric (a per-task parameter cap is enforced before each test)
- Hyperparameter tuning (learning rates, batch sizes, etc.)
- Copying a reference baseline with cosmetic changes
- Pure engineering tricks without algorithmic novelty

Parameter count is capped (enforced before each test); architectural changes within that budget are encouraged.

IMPORTANT workflow:
1. FIRST call edit() to implement your improved algorithm. Do NOT call test() before making edits.
2. THEN call test() to run the experiment. Each run is numbered (#1, #2, ...).
3. Review the metrics, then edit() to improve your solution based on the feedback.
4. Call test() again to verify the improvement. You MUST iterate at least once \
(edit → test → review → edit → test) before submitting, unless only 1 test is allowed.
5. When satisfied, call submit(n=N) to submit your best test #N as final.
You have a limited number of test() calls, so make each one count by editing first.

Available tools:
- edit(op, filename, ...): Modify files in the workspace. Choose op by how
  much you are changing:
  - op='rewrite': replace the WHOLE editable region with `content`. This is
    the normal way to write your solution. It needs no line numbers and no
    copy of the old code, so it cannot fail with 'not found' or 'outside the
    editable range'. Prefer it whenever you are (re)implementing the stub.
  - op='str_replace': replace the single, unique occurrence of old_str with
    new_str — for SMALL surgical fixes only. Keep old_str to 1–10 lines. Never
    paste a whole function, class or docstring as old_str; that is what
    op='rewrite' is for, and long anchors almost always fail to match.
  - op='create': create a new file (only if allow_create=true)
- view(filename, start_line=None, end_line=None): Read the file back. Call
  it before str_replace whenever you are unsure of the exact text, and
  always after an undo().

  Reading the file correctly matters more than anything else here:
  * Every successful edit echoes back the file's CURRENT editable region.
    That echo is the truth. The listing in this first message is only true
    until your first edit — after that, do not copy old_str out of it.
  * The 'NNNNNN: ' line numbers in every listing are a display prefix. Never
    include them in old_str or in content.
  * If an edit is rejected, read the error: it quotes the nearest real text
    in the file. Anchor on that, or switch to op='rewrite'.
- test(): Run a new experiment. Executes training and evaluation. Each run is
  numbered #1, #2, etc. The first test runs all seeds; intermediate tests run one seed.
  You have a limited budget of test() calls, so make each one count by editing first.
  If max tests is reached, the last test is auto-submitted.
- submit(n=N): Submit the result from test #N as your final answer (1-indexed).
  This does NOT re-run anything — it picks a previous result. Use n=-1 for the latest.
  You must call test() at least once before you can submit.
- undo(n=1): Revert the last n edit operations.

Constraints:
- Each file shown in the prompt is labeled [READ-ONLY] or [EDITABLE — lines X–Y only].
  Only edit files and line ranges marked EDITABLE. Do not touch READ-ONLY files.
- When a file has multiple editable regions, editing one region may shift line numbers \
in subsequent regions. Edit from the last (bottom-most) region first, or check the \
updated editable ranges shown after each edit.
- You MUST call test() at least once before you can call submit().
- When you are done, call submit(n=N) to submit your best test result.
- If your algorithm requires new hyperparameters (e.g., cql_alpha, expectile_tau) that are not
  in the existing config, hardcode their values directly in your code (e.g., in __init__).
  You cannot modify the training loop or config to pass them via command line.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "edit",
        "description": (
            "Edit files in the workspace. Pick the operation by HOW MUCH you are changing:\n"
            "\n"
            "  rewrite: Replace the WHOLE editable region of the file with `content`.\n"
            "    Use this to implement or re-implement the stub you were given — it is the\n"
            "    normal way to write your solution. It needs NO line numbers and NO copy of\n"
            "    the old code, so it cannot fail with 'not found' or 'outside the editable\n"
            "    range'. `content` is your complete new code for that region.\n"
            "\n"
            "  str_replace: Replace the SINGLE, UNIQUE occurrence of `old_str` with `new_str`.\n"
            "    Use this only for a SMALL, surgical change (fix one line, tweak a constant).\n"
            "    `old_str` must reproduce text that is in the file right now and must be\n"
            "    unique. Keep it SHORT — 1 to 10 lines. Do NOT paste a whole function or\n"
            "    docstring as `old_str`; if the change is that big, use op='rewrite'.\n"
            "    An empty `new_str` deletes `old_str`.\n"
            "\n"
            "  create: Create a NEW file with `content`. Only available if allow_create=true.\n"
            "\n"
            "After every successful edit this tool echoes the file's current editable region\n"
            "back to you. That echo — not the copy in the first message — is the truth about\n"
            "what the file now contains. Anchor the next str_replace on it, or call view().\n"
            "File paths are relative to the package root (e.g. 'LLaMA-Factory/src/...').\n"
            "Lines outside the editable region are protected and must NOT be modified."),
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string", "enum": ["rewrite", "str_replace", "create"],
                   "description": (
                       "The edit operation, with its REQUIRED companion arguments:\n"
                       "  'rewrite'     -> requires content   (replaces the whole editable region)\n"
                       "  'str_replace' -> requires old_str AND new_str (small surgical change)\n"
                       "  'create'      -> requires content   (new file)")},
            "filename": {"type": "string",
                         "description": "Package-relative path to the file (e.g. 'pytorch-vision/custom_loss.py')."},
            "content": {"type": "string",
                        "description": (
                            "Your complete new code. Required for op='rewrite' (becomes the entire "
                            "editable region) and for op='create'.")},
            "old_str": {"type": "string",
                        "description": (
                            "Text to replace (required for op='str_replace'). Must appear in the file's "
                            "CURRENT contents and occur exactly once. Keep it to 1-10 lines; use "
                            "op='rewrite' for anything larger. Never include the 'NNNNNN: ' line-number "
                            "prefixes that appear in the displayed code.")},
            "new_str": {"type": "string",
                        "description": "Replacement text (required for op='str_replace'). An empty string deletes `old_str`."}},
            "required": ["op", "filename"]}}},
    {"type": "function", "function": {
        "name": "view",
        "description": (
            "Read the current contents of a file in the workspace. Call this before "
            "op='str_replace' whenever you are unsure of the exact text (especially "
            "after an edit, an undo, or when the initial prompt has gone stale) — "
            "old_str must reproduce the file's text, and this is how you check it.\n"
            "Line numbers in the output are a display prefix ('NNNNNN: ') and must "
            "NOT be included in old_str.\n"
            "With no line range, shows the editable region of the file."),
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string",
                         "description": "Package-relative path (e.g. 'scikit-learn/custom_calibration.py')."},
            "start_line": {"type": "integer", "description": "First line to show (1-indexed, optional)."},
            "end_line": {"type": "integer", "description": "Last line to show (1-indexed, inclusive, optional)."}},
            "required": ["filename"]}}},
    {"type": "function", "function": {
        "name": "test",
        "description": (
            "Run a new experiment. Executes training and evaluation, then returns metrics. "
            "Each run is numbered #1, #2, etc. All runs use all configured seeds. "
            "You have a limited test budget."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": (
            "Submit a previous test result as your final answer. This does NOT re-run "
            "anything — it selects a result you already obtained. You must have run "
            "test() at least once before calling submit()."),
        "parameters": {"type": "object", "properties": {
            "n": {"type": "integer",
                  "description": "The test number to submit (1-indexed). e.g. n=1 submits the result from test #1."}},
            "required": ["n"]}}},
    {"type": "function", "function": {
        "name": "undo",
        "description": ("Revert the last n file modification actions (create/insert/replace) by "
                        "restoring pre-edit snapshots. Does not undo test calls."),
        "parameters": {"type": "object", "properties": {
            "n": {"type": "integer", "description": "Number of edit actions to undo (default: 1)."}}}}},
]


def read(p):
    return open(p, encoding='utf-8').read()


def largest_fence(text):
    best = ''
    for _lang, body in FENCE.findall(text):
        if len(body) > len(best):
            best = body
    return best.rstrip('\n') + '\n' if best else ''


# ---------------------------------------------------------------------------
# str_replace op decomposition: difflib opcodes -> unique-match ops, replayed.
# ---------------------------------------------------------------------------
def _count(hay, needle):
    n = i = 0
    while True:
        i = hay.find(needle, i)
        if i < 0:
            return n
        n += 1
        i += 1  # overlapping occurrences count as ambiguous too


def decompose(old, new):
    """Return a list of (old_str, new_str) unique-match ops turning old into new.

    Grows context around each changed block until old_str is unique in the
    CURRENT state; merges blocks whose context would overlap. Raises on failure.
    """
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    blocks = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != 'equal']
    if not blocks:
        return []
    # merge blocks separated by < 3 equal lines (context would collide)
    merged = [list(blocks[0])]
    for i1, i2, j1, j2 in blocks[1:]:
        if i1 - merged[-1][1] < 3:
            merged[-1][1], merged[-1][3] = i2, j2
        else:
            merged.append([i1, i2, j1, j2])

    ops, cur = [], old
    # apply back-to-front so earlier line indices stay valid in `a`
    for i1, i2, j1, j2 in reversed(merged):
        for ctx in range(0, 40):
            lo, hi = max(0, i1 - ctx), min(len(a), i2 + ctx)
            old_str = ''.join(a[lo:hi])
            if old_str and _count(cur, old_str) == 1:
                new_str = ''.join(a[lo:i1] + b[j1:j2] + a[i2:hi])
                cur = cur.replace(old_str, new_str, 1)
                ops.append((old_str, new_str))
                break
        else:
            raise ValueError('no unique anchor for block at lines %d-%d' % (i1 + 1, i2))
    if cur != new:
        raise ValueError('replay mismatch after op decomposition')
    ops.reverse()  # present in top-of-file-first order? keep application order instead
    # NOTE: ops were applied back-to-front; re-verify in the recorded (reversed) order
    cur2 = old
    final_ops = []
    for old_str, new_str in reversed(ops):
        if _count(cur2, old_str) != 1:
            raise ValueError('order-dependent anchor collision')
        cur2 = cur2.replace(old_str, new_str, 1)
        final_ops.append((old_str, new_str))
    if cur2 != new:
        raise ValueError('replay mismatch in recorded order')
    return final_ops


# Live-contract op policy. The system prompt is explicit: rewrite is "the normal
# way to write your solution", str_replace is for SMALL fixes with a 1-10 line
# anchor. A rung transition therefore becomes str_replace ops ONLY when every
# changed block is small AND every unique anchor fits in 10 lines; otherwise it
# is ONE op='rewrite' carrying the full new editable region (= the whole file
# here). This also retires the anti-prompt giant-old_str ops of the first cut.
STR_REPLACE_MAX_ANCHOR_LINES = 10
STR_REPLACE_MAX_NEW_LINES = 12
STR_REPLACE_MAX_OPS = 6


def plan_ops(prev, new):
    """Return the rung's edit ops as dicts: str_replace chain or a single rewrite."""
    if prev is None:
        return [{'op': 'create', 'content': new}]
    try:
        pairs = decompose(prev, new)
    except ValueError:
        return [{'op': 'rewrite', 'content': new}]
    if not pairs:
        raise ValueError('no-op transition (identical states)')
    if len(pairs) > STR_REPLACE_MAX_OPS:
        return [{'op': 'rewrite', 'content': new}]
    for old_str, new_str in pairs:
        if (old_str.rstrip('\n').count('\n') + 1 > STR_REPLACE_MAX_ANCHOR_LINES
                or new_str.rstrip('\n').count('\n') + 1 > STR_REPLACE_MAX_NEW_LINES):
            return [{'op': 'rewrite', 'content': new}]
    return [{'op': 'str_replace', 'old_str': o, 'new_str': n} for o, n in pairs]


def line_span(state, old_str):
    idx = state.find(old_str)
    start = state.count('\n', 0, idx) + 1
    end = start + old_str.rstrip('\n').count('\n')
    return start, end


# ---------------------------------------------------------------------------
# Post-edit echo — faithful clone of tools.py _file_snapshot() for a file whose
# editable region is the entire file (ranges [(1, N)]): full listing up to 160
# lines, head/tail peek with a view() elision line beyond that, 6000-char cut
# on a line boundary with the view() truncation notice.
# ---------------------------------------------------------------------------
SNAPSHOT_MAX_LINES = 160
SNAPSHOT_MAX_CHARS = 6000


def file_echo(filename, state):
    all_lines = state.splitlines()
    n = len(all_lines)
    if n == 0:
        return ''
    header = (
        '[Current contents of %s | editable: 1–%d | total: %d lines]\n'
        'This is the file as it is NOW — use it, not the copy in the first '
        'message, when choosing old_str. The \'NNNNNN: \' prefixes are display '
        'only; never put them in old_str.' % (filename, n, n)
    )
    lines_out = []
    if n <= SNAPSHOT_MAX_LINES:
        for i in range(1, n + 1):
            lines_out.append('%6d: %s' % (i, all_lines[i - 1]))
    else:
        peek = max(4, SNAPSHOT_MAX_LINES // 2)
        if n <= peek * 2 + 2:
            for i in range(1, n + 1):
                lines_out.append('%6d: %s' % (i, all_lines[i - 1]))
        else:
            for i in range(1, peek + 1):
                lines_out.append('%6d: %s' % (i, all_lines[i - 1]))
            lines_out.append(
                "       ... (%d more lines not shown — call "
                "view(filename='%s', start_line=..., end_line=...) "
                "to read them) ..." % (n - peek * 2, filename))
            for i in range(n - peek + 1, n + 1):
                lines_out.append('%6d: %s' % (i, all_lines[i - 1]))
    out = header + '\n' + '\n'.join(lines_out)
    if len(out) > SNAPSHOT_MAX_CHARS:
        out = out[:SNAPSHOT_MAX_CHARS].rsplit('\n', 1)[0] + (
            "\n... (echo truncated at %d chars — call "
            "view(filename='%s', start_line=..., end_line=...) for the rest)"
            % (SNAPSHOT_MAX_CHARS, filename))
    return out


# ---------------------------------------------------------------------------
# Filename resolution
# ---------------------------------------------------------------------------
# Hand-set workspace paths where extraction gets it wrong: repo-internal path
# leaks (the trained channel must never carry methods/-/results/ paths — see the
# 2026-07-25 trained-channel-leak cleanup) and non-Python ladders whose
# synthesized name would get a .py extension. Paths mirror the upstream repo
# layout each ladder is actually about.
FILENAME_OVERRIDES = {
    'bio-scrna-denoise': 'denoising/denoise_ttt.py',
    'convnext-modernization': 'ConvNeXt/models/convnext.py',
    'sys-1brc-speedrun': 'onebrc/CalculateAverage_custom.java',
    'sys-flash-attention-speedrun': 'flash_attn/flash_fwd_kernel.cu',
    'sys-llamacpp-quantization': 'llama.cpp/ggml-quants.c',
    'sys-llmc-gpt2-speedrun': 'llm.c/train_gpt2.c',
}
_LEAK_PREFIX = re.compile(r'^(methods/[^/]+/code/|results/|trajectories/[^/]+/|data_v4/[^/]+/)')


def resolve_filename(task_dir, task):
    if task in FILENAME_OVERRIDES:
        return FILENAME_OVERRIDES[task], 'override'
    skel = os.path.join(task_dir, 'agentic_skeleton.txt')
    if os.path.isfile(skel):
        m = re.search(r'file="([^"]+)"', open(skel, encoding='utf-8').read(400))
        if m:
            return m.group(1), 'june_skeleton'
    # search meta notes / initial context for a named .py edit surface
    meta = json.load(open(os.path.join(task_dir, 'meta.json')))
    hay = meta.get('notes', '') + ' ' + meta.get('ranking_basis', '')
    m = re.search(r'([A-Za-z0-9_./-]+\.py)\b', hay)
    if m and '/' in m.group(1):
        return _LEAK_PREFIX.sub('', m.group(1)), 'meta_notes'
    if m:
        return 'workspace/' + m.group(1).lstrip('./'), 'meta_notes_bare'
    slug = re.sub(r'[^a-z0-9]+', '_', task.lower()).strip('_')
    return 'workspace/custom_%s.py' % slug, 'synthesized'


_EXT_LANG = {'.py': 'python', '.java': 'java', '.c': 'c', '.cu': 'cuda', '.cpp': 'cpp',
             '.h': 'c', '.yaml': 'yaml', '.yml': 'yaml', '.sh': 'bash', '.jl': 'julia',
             '.rs': 'rust'}


def fence_lang(filename):
    return _EXT_LANG.get(os.path.splitext(filename)[1].lower(), 'python')


# ---------------------------------------------------------------------------
# Episode assembly
# ---------------------------------------------------------------------------
def numbered(code):
    return '\n'.join('%6d: %s' % (i + 1, ln) for i, ln in enumerate(code.splitlines()))


# Cumulative-stack ladders: the ladder accumulates technologies (each rung's
# measured number inherits the previous rungs' components — airbench additive
# speedups, nanoGPT record chain, vLLM/llm.c stacks, RoBERTa recipe). For those,
# a rewrite/str_replace op sequence CONTRADICTS the metrics (it would erase the
# previous technique while the next number claims to build on it). Their episodes
# instead emit one edit(op='create') per rung, one new module file per technique
# — the workspace only grows, matching the stacking semantics. The module path
# is derived from the trajectory's own workspace filename + the rung slug.
CUMULATIVE_TASKS = {
    'cv-cifar10-speedrun', 'dl-resnet-imagenet-speedup', 'lm-hlb-gpt-speedrun',
    'lm-nanochat-pipeline', 'lm-nanogpt-speedrun', 'sys-gptfast-inference-speedrun',
    'sys-llm-serving-throughput', 'sys-llmc-gpt2-speedrun', 'roberta-pretraining-recipe',
}


def module_path(filename, slug):
    """workspace/custom_speedrun.py + patch-whitening -> workspace/patch_whitening.py
    legacy/airbench96.py -> legacy/patch_whitening.py"""
    parts = filename.rsplit('/', 1)
    base = parts[0] if len(parts) > 1 else ''
    ext = os.path.splitext(parts[-1])[1] or '.py'
    return (base + '/' if base else '') + slug.replace('-', '_') + ext


def initial_prompt(context_md, filename, code, n_actions, n_tests, baseline=None,
                   cumulative=False):
    parts = [context_md.rstrip()]
    if code:
        parts.append('\n## %s  [EDITABLE — entire file only]\n```%s\n%s\n```'
                     % (filename, fence_lang(filename), numbered(code)))
    else:
        parts.append('\n## %s\n(The file does not exist yet — your first edit must be '
                     "edit(op='create', filename='%s', content=...).)" % (filename, filename))
    if baseline:
        slug, fb = baseline
        parts.append('\n## Baseline results (the code above, already measured — '
                     'reference `%s`)\n\n%s' % (slug, fb))
    if cumulative:
        parts.append('\nThe workspace grows module by module: implement each new technique '
                     'as a NEW file with edit(op=\'create\', ...) next to the code shown '
                     'above. Do not rewrite or delete the existing files — the measured '
                     'reference stack builds on them.')
    budget = [
        '- **Action budget**: %d total tool calls '
        '(every edit / test / undo / web_search / web_extract counts; submit does not)' % n_actions,
        '- **Test invocations**: at most %d '
        '(each test() call also consumes one action from the budget above)' % n_tests,
    ]
    if n_tests >= 2:
        budget.append('- You **must** iterate at least once '
                      '(edit → test → review → edit → test) before submitting.')
    parts.append('\n## Your Budget\n' + '\n'.join(budget))
    return '\n'.join(parts)


def asst(think_slot, say, name, args):
    return {'role': 'assistant', 'reasoning_content': think_slot, 'content': say,
            'tool_calls': [{'type': 'function',
                            'function': {'name': name, 'arguments': args}}]}


def tool(text):
    return {'role': 'tool', 'content': text}


def build_task(task_dir, task):
    meta = json.load(open(os.path.join(task_dir, 'meta.json')))
    steps = [s for s in sorted(meta.get('steps', []), key=lambda s: s.get('n', 0))
             if s.get('answer')]
    if task in TYPE1_FINALE:
        steps = [s for s in steps if not s.get('finale')]
    # An episode rung IS a measured test() — a trailing rung with no feedback file
    # (the documented-ladder `finale` endpoints) has no metrics to return, and we do
    # not fabricate numbers. It stays out of the agentic episode (still covered by
    # the trajectory pipeline); the episode ends on the last measured rung.
    while steps and not (steps[-1].get('feedback')
                         and os.path.isfile(os.path.join(task_dir, steps[-1]['feedback']))):
        steps = steps[:-1]
    if len(steps) < 2:
        raise ValueError('fewer than 2 usable rungs (submit needs >=2 tests)')

    states, fbs = [], []
    for s in steps:
        code = largest_fence(read(os.path.join(task_dir, s['answer'])))
        if len(code) < 80:
            raise ValueError('rung %s: no usable code fence in %s' % (s.get('slug'), s['answer']))
        states.append(code)
        fb = s.get('feedback')
        fbp = os.path.join(task_dir, fb) if fb else None
        if not (fbp and os.path.isfile(fbp)):
            raise ValueError('rung %s: missing feedback (test() needs real metrics)' % s.get('slug'))
        fbs.append(read(fbp).strip())

    init_f = meta.get('initial_context_file', '00-initial-context.md')
    context_md = read(os.path.join(task_dir, init_f)).strip()
    scaffold = largest_fence(context_md)
    # scaffold must genuinely be a prior state of the same file, else start from create
    use_scaffold = bool(scaffold) and difflib.SequenceMatcher(
        a=scaffold, b=states[0], autojunk=False).quick_ratio() > 0.3

    # When the context scaffold IS rung 1 up to cosmetics (18 tasks), diffing them
    # yields decorative comment/whitespace ops — the "no-op opener" the 2026-08
    # bloat audit condemned. The truthful episode shape there is the real MLS one:
    # the prompt shows the rung-1 baseline verbatim WITH its measured numbers, and
    # the agent's first action is rung 2's design edit.
    baseline_block = None
    if use_scaffold and len(steps) >= 3 and difflib.SequenceMatcher(
            a=scaffold, b=states[0], autojunk=False).ratio() > 0.9:
        scaffold = states[0]
        baseline_block = (steps[0].get('slug', 'baseline'), fbs[0])
        steps, states, fbs = steps[1:], states[1:], fbs[1:]

    filename, fname_src = resolve_filename(task_dir, task)

    # ---- plan all rung transitions first (fail before writing anything)
    cumulative = task in CUMULATIVE_TASKS
    rung_ops = []
    prev = scaffold if use_scaffold else None
    for ri, state in enumerate(states):
        if cumulative:
            rung_ops.append([{'op': 'create',
                              'filename': module_path(filename, steps[ri].get('slug', 'rung%d' % (ri + 1))),
                              'content': state}])
        else:
            try:
                rung_ops.append(plan_ops(prev, state))
            except ValueError as e:
                raise ValueError('rung %d: %s' % (ri + 1, e))
        prev = state

    n_edits = sum(len(o) for o in rung_ops)
    n_tests = len(steps)
    n_actions = n_edits + n_tests

    msgs = [{'role': 'user',
             'content': initial_prompt(context_md, filename,
                                       scaffold if use_scaffold else '',
                                       n_actions, n_tests, baseline=baseline_block,
                                       cumulative=cumulative)}]
    seq = 0  # unique per assistant turn -> fills files key on the full sentinel

    def next_seq():
        nonlocal seq
        seq += 1
        return seq

    replay = scaffold if use_scaffold else None
    for ri, (s, ops, fb) in enumerate(zip(steps, rung_ops, fbs)):
        slug = s.get('slug', 'rung%d' % (ri + 1))
        for oi, op in enumerate(ops):
            kind = 'design' if oi == 0 else 'followup'
            n = next_seq()
            slot = '[[THINK kind=%s rung=%d slug=%s seq=%d]]' % (kind, ri + 1, slug, n)
            say = '[[SAY rung=%d slug=%s seq=%d]]' % (ri + 1, slug, n) if oi == 0 else ''
            if op['op'] == 'create':
                fname = op.get('filename', filename)
                msgs.append(asst(slot, say, 'edit',
                                 {'op': 'create', 'filename': fname,
                                  'content': op['content']}))
                result = 'Created: %s' % fname
                if not cumulative:      # cumulative stacks grow files; replace mode
                    replay = op['content']  # replaces the single-file state
            elif op['op'] == 'rewrite':
                old_n = len(replay.splitlines())
                new_n = len(op['content'].splitlines())
                msgs.append(asst(slot, say, 'edit',
                                 {'op': 'rewrite', 'filename': filename,
                                  'content': op['content']}))
                replay = op['content']
                result = ('OK: Rewrote the editable region of %s — replaced lines '
                          '1..%d (%d lines) with %d line(s). '
                          'Editable range: entire file.' % (filename, old_n, old_n, new_n))
            else:
                a, b = line_span(replay, op['old_str'])
                msgs.append(asst(slot, say, 'edit',
                                 {'op': 'str_replace', 'filename': filename,
                                  'old_str': op['old_str'], 'new_str': op['new_str']}))
                replay = replay.replace(op['old_str'], op['new_str'], 1)
                result = ('OK: Replaced 1 occurrence in %s (lines %d..%d). '
                          'Editable range: entire file.' % (filename, a, b))
            echo = file_echo(filename, replay) if not cumulative else ''
            msgs.append(tool(result + ('\n\n' + echo if echo else '')))
        if not cumulative and replay != states[ri]:
            raise ValueError('rung %d: replay diverged from target state' % (ri + 1))
        msgs.append(asst('[[THINK kind=pre_test rung=%d slug=%s seq=%d]]'
                         % (ri + 1, slug, next_seq()), '', 'test', {}))
        remaining = n_tests - (ri + 1)
        header = ('[Test #%d] (%d test%s remaining; call submit(n=N) to choose which '
                  'test result to submit as final)\n\n'
                  % (ri + 1, remaining, 's' if remaining != 1 else ''))
        if remaining == 0:
            header += ('[NOTE] This was your last test. You MUST now call submit(n=X) '
                       'to choose which test result to submit as your final answer.\n\n')
        msgs.append(tool(header + fb))

    best = n_tests  # ladders ascend; type1 finale already stripped. Reviewed via report.
    msgs.append(asst('[[THINK kind=submit best=%d seq=%d]]' % (best, next_seq()),
                     '', 'submit', {'n': best}))
    msgs.append(tool('[submit] Submitting result from test #%d as final.\n\n%s'
                     % (best, fbs[best - 1])))

    out = {
        'task': task,
        'schema': 'agentic_v2',
        'harness': 'mlsbench-devfix-rewrite-2026-08-24',
        'file': filename,
        'filename_source': fname_src,
        'year': None,  # filled from trajectories.json at SFT build time, as before
        'scaffold_from_context': use_scaffold,
        'cumulative_stack': cumulative,
        'baseline_in_prompt': baseline_block[0] if baseline_block else None,
        'n_rungs': n_tests,
        'n_edit_ops': n_edits,
        'best_test': best,
        'system': SYSTEM_PROMPT,
        'tools': TOOLS,
        'messages': msgs,
    }
    return out


# ---------------------------------------------------------------------------
# Lint mode: after the prose workflow, every slot must be replaced by real text.
# ---------------------------------------------------------------------------
SENTINEL = re.compile(r'\[\[(THINK|SAY)[^\]]*\]\]')
# The prompt's numbered [EDITABLE] file dump IS the replay start state for every
# scaffold_from_context task (148 of 164) — those episodes open on a str_replace,
# never a create, so seeding the lint's replay from `None` failed all of them.
PROMPT_DUMP = re.compile(r'^## \S+  \[EDITABLE — entire file(?: only)?\]\n```[a-zA-Z0-9_+-]*\n(.*?)\n```',
                         re.S | re.M)
LINENO = re.compile(r'^ *\d+: ')


def seed_replay(messages):
    """Recover the scaffold the builder started `replay` from (strip '%6d: ' prefixes)."""
    if not messages or messages[0].get('role') != 'user':
        return None
    m = PROMPT_DUMP.search(messages[0].get('content') or '')
    if not m:
        return None
    return '\n'.join(LINENO.sub('', ln, count=1) for ln in m.group(1).split('\n')) + '\n'


def lint(paths):
    bad = 0
    for p in paths:
        d = json.load(open(p))
        replay = seed_replay(d.get('messages'))
        for i, m in enumerate(d['messages']):
            if m['role'] != 'assistant':
                continue
            rc = (m.get('reasoning_content') or '').strip()
            if not rc:
                print('%s: msg %d has EMPTY reasoning_content' % (p, i)); bad += 1
            if SENTINEL.search(rc) or SENTINEL.search(m.get('content') or ''):
                print('%s: msg %d still carries a [[..]] sentinel' % (p, i)); bad += 1
            fn = m['tool_calls'][0]['function']
            if fn['name'] == 'edit':
                args = fn['arguments']
                if args['op'] in ('create', 'rewrite'):
                    replay = args['content']
                else:
                    if replay is None or _count(replay, args['old_str']) != 1:
                        print('%s: msg %d old_str not unique in replay state' % (p, i)); bad += 1
                    else:
                        replay = replay.replace(args['old_str'], args['new_str'], 1)
    print('lint: %d problem(s) in %d file(s)' % (bad, len(paths)))
    return bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if '--lint' in sys.argv:
        # lint targets the FILLED files (the training input); falls back to the
        # skeleton when a task has not been filled yet (then sentinels flag it).
        def _p(t):
            f = os.path.join('trajectories', t, 'agentic_v2_filled.json')
            return f if os.path.isfile(f) else os.path.join('trajectories', t, OUT_NAME)
        paths = [_p(t) for t in args] if args \
            else sorted(glob.glob('trajectories/*/agentic_v2_filled.json'))
        sys.exit(1 if lint(paths) else 0)

    tasks = args or sorted(os.path.basename(os.path.dirname(p))
                           for p in glob.glob('trajectories/*/meta.json'))
    report = {'built': [], 'skipped': [], 'failed': []}
    for task in tasks:
        task_dir = os.path.join('trajectories', task)
        if task in DROP_TRAJ:
            report['skipped'].append({'task': task, 'reason': 'decontam drop_traj'})
            continue
        try:
            out = build_task(task_dir, task)
        except Exception as e:
            report['failed'].append({'task': task, 'reason': str(e)})
            continue
        with open(os.path.join(task_dir, OUT_NAME), 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        report['built'].append({'task': task, 'file': out['file'],
                                'filename_source': out['filename_source'],
                                'rungs': out['n_rungs'], 'edit_ops': out['n_edit_ops'],
                                'scaffold': out['scaffold_from_context'],
                                'baseline_in_prompt': out['baseline_in_prompt'],
                                'best_test': out['best_test']})
    with open(REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print('built %d, skipped %d, failed %d -> %s'
          % (len(report['built']), len(report['skipped']), len(report['failed']), REPORT))
    for r in report['failed']:
        print('  FAIL %s: %s' % (r['task'], r['reason']))


if __name__ == '__main__':
    main()
