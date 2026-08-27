#!/usr/bin/env python3
"""Real-template mode for agentic v2 episodes (MLS-Bench tasks).

2026-08-26 user ruling: the episode's INPUT must be what the harness actually shows —
the task's editable file(s) as they stand after pre_edit + mid_edit (the real
template, full listing, `[EDITABLE — lines a–b only]`), not the rung's code snippet
posing as a whole-file. Every edit is executed through the REAL
`mlsbench.agent.tools.WorkspaceTools` on a throw-away workspace, so the tool result
strings, the post-edit echo, the shifting `Editable range` and the syntax gate are
the harness's own, not a clone. Rung 1 replaces the template's editable region with
the rung-1 code; rung k+1 replaces the previous rung's region content (or a small
str_replace chain when the diff is small). Non-MLS ladders keep the legacy
single-file framing in build_agentic_v2.py.
"""
import contextlib
import difflib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = 'training/FrontierSmith/.cache/mlsbench-eval'
FENCE = re.compile(r'```([a-zA-Z0-9_+-]*)\n(.*?)```', re.S)
_EXT_LANG = {'.py': 'python', '.yaml': 'yaml', '.yml': 'yaml', '.sh': 'bash', '.json': 'json',
             '.jsonl': 'json', '.toml': 'toml', '.cfg': 'ini', '.txt': 'text'}
_T = None


def harness():
    global _T
    if _T is None:
        sys.path.insert(0, ROOT + '/src')
        os.environ['MLSBENCH_REWRITE_OP'] = '0'
        os.environ.pop('MLSBENCH_STRICT_STR_REPLACE', None)
        from mlsbench.agent import tools as T
        _T = T
    return _T


def has_config(task):
    return os.path.isfile(f'{ROOT}/tasks/{task}/config.json')


def _norm(s):
    return s.lower().replace('-', '').replace('_', '')


def _purge_task_modules():
    """mid_edit.py helpers (dgp, labels, ...) share names across tasks; drop any
    module that lives under tasks/ or a throw-away workspace so imports resolve fresh."""
    tasks_dir = os.path.abspath(f'{ROOT}/tasks')
    for k, m in list(sys.modules.items()):
        f = getattr(m, '__file__', None) or ''
        if f and (os.path.abspath(f).startswith(tasks_dir) or '/agv2mls_' in f):
            sys.modules.pop(k, None)
    importlib.invalidate_caches()


class Sim:
    """A real WorkspaceTools over a temp workspace holding only the task's declared files."""

    def __init__(self, task, max_tests):
        T = harness()
        self.task = task
        self.cfg = json.load(open(f'{ROOT}/tasks/{task}/config.json'))
        self.tmp = tempfile.mkdtemp(prefix='agv2mls_')
        self.tools = T.WorkspaceTools(task, self.cfg, self.cfg['files'], self.tmp, ROOT, max_tests)
        self.tools.use_replace = True       # str_replace contract (interactive.py does the same)
        self.tools.allow_rewrite = False
        self.tools.syntax_gate = True       # reject-and-revert on a new syntax error
        self.wtd = Path(self.tools.workspace_task_dir)
        self.missing = []
        self._materialize()

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- workspace -----------------------------------------------------------
    def _ext_pkg_dir(self, pkg):
        base = Path(ROOT) / 'vendor' / 'external_packages'
        for d in base.iterdir():
            if d.is_dir() and _norm(d.name) == _norm(pkg):
                return d
        return None

    def _materialize(self):
        T = harness()
        needed = [f['filename'] for f in self.cfg['files']]
        for fn in needed:
            pkg, rest = fn.split('/', 1) if '/' in fn else (fn, '')
            if (Path(ROOT) / 'tasks' / self.task / fn).is_file() and self._ext_pkg_dir(pkg) is None:
                continue                      # task-relative file: resolves via tasks/<task>/
            (self.wtd / pkg).mkdir(parents=True, exist_ok=True)
            src_dir = self._ext_pkg_dir(pkg)
            src = (src_dir / rest) if (src_dir and rest) else None
            if src and src.is_file():
                dst = self.wtd / pkg / rest
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
        pre = T.load_pre_edit_ops(self.cfg, Path(ROOT) / 'vendor' / 'pkg_configs')
        # mid_edit.py files import sibling helpers (dgp, labels, ...) under colliding
        # names and sometimes need the task package's own conda env: load them in a
        # fresh subprocess (base python first, then envs/mlsbench-<pkg>/bin/python)
        mid = self._load_mid_ops_isolated()
        want = {_norm(f) for f in needed}
        ops = []
        for o in pre + mid:
            if _norm(o.get('file', '')) not in want:
                continue
            if o.get('op') != 'create':
                try:
                    p = self.tools._resolve_workspace_path(o['file'])
                except FileNotFoundError:
                    p = None
                if p is None or not p.exists():
                    self.missing.append((o['file'], o['op']))
                    continue
            ops.append(o)
        with contextlib.redirect_stdout(io.StringIO()):
            self.tools.apply_pre_edit(ops)
        rigorous = self.cfg.get('rigorous_codebase', False) and bool(self.cfg.get('baselines'))
        for entry in self.cfg['files']:
            fn = entry['filename']
            shown = bool(entry.get('read')) and (entry.get('edit') or not rigorous)
            if not entry.get('edit') and not shown:
                continue                  # read-only file the rigorous prompt never lists
            try:
                p = self.tools._resolve_workspace_path(fn)
                ok = p.exists()
            except FileNotFoundError:
                ok = False
            if not ok:
                self.missing.append((fn, 'absent'))

    def _load_mid_ops_isolated(self):
        import subprocess
        code = ('import sys, json; sys.path.insert(0, %r); '
                'from mlsbench.agent.tools import load_mid_edit_ops; from pathlib import Path; '
                'ops = load_mid_edit_ops(%r, Path(%r)); '
                'sys.stdout.write("\\n@@OPS@@" + json.dumps(ops))'
                % (os.path.abspath(ROOT + '/src'), self.task, os.path.abspath(ROOT + '/tasks')))
        pythons = [sys.executable]
        pkgs = {e.get('package') for e in self.cfg.get('test_cmds', []) if e.get('package')}
        for pk in sorted(pkgs):
            for d in Path(os.path.expanduser('~/miniconda3/envs')).glob('mlsbench-*'):
                if _norm(d.name[len('mlsbench-'):]) == _norm(pk) and (d / 'bin' / 'python').exists():
                    pythons.append(str(d / 'bin' / 'python'))
        last = None
        for py in pythons:
            r = subprocess.run([py, '-c', code], capture_output=True, text=True, timeout=600,
                               cwd=os.getcwd())
            if r.returncode == 0 and '@@OPS@@' in r.stdout:
                return json.loads(r.stdout.rsplit('@@OPS@@', 1)[1])
            last = (r.stderr or r.stdout).strip().splitlines()[-1:] if (r.stderr or r.stdout) else ['?']
        raise RuntimeError('mid_edit load failed in all interpreters: %s' % (last,))

    # -- file access -----------------------------------------------------------
    def read(self, fn):
        return self.tools._resolve_workspace_path(fn).read_text()

    def editable_files(self):
        return [f['filename'] for f in self.cfg['files'] if f.get('edit')]

    def regions(self, fn):
        """[(start, end, text)] editable regions of fn as the harness computes them NOW."""
        text = self.read(fn)
        lines = text.splitlines(True)
        out = []
        for s, e in self.tools._compute_editable_ranges(fn, len(lines)):
            s_c = max(1, min(s, len(lines)))
            e_c = max(s_c - 1, min(e, len(lines)))
            out.append((s_c, e_c, ''.join(lines[s_c - 1:e_c])))
        return out

    def edit(self, fn, old_str, new_str):
        return self.tools.edit('str_replace', fn, old_str=old_str, new_str=new_str)

    # -- prompt pieces (verbatim ports of base.py build_initial_prompt) --------
    def file_sections(self):
        cfg = self.cfg
        rigorous = cfg.get('rigorous_codebase', False)
        baselines = cfg.get('baselines', {})
        sections = []
        for entry in cfg['files']:
            filename = entry['filename']
            read_ranges = entry.get('read', [])
            if not read_ranges:
                continue
            editable = 'edit' in entry
            if rigorous and baselines and not editable:
                continue
            edit_ranges = entry.get('edit', [])
            if not editable or not edit_ranges:
                edit_note = 'READ-ONLY — do not edit'
            else:
                range_strs = ['entire file' if r['start'] == -1 else f"lines {r['start']}–{r['end']}"
                              for r in edit_ranges]
                edit_note = f"EDITABLE — {', '.join(range_strs)} only"
            try:
                path = self.tools._resolve_workspace_path(filename)
                all_lines = path.read_text().splitlines()
            except Exception as exc:
                sections.append(f'\n## {filename}  [{edit_note}]\n(could not read: {exc})')
                continue
            file_sections = []
            for rng in read_ranges:
                start, end = rng['start'], rng['end']
                if start == -1 and end == -1:
                    file_sections.append('\n'.join(f'{i + 1:6d}: {line}' for i, line in enumerate(all_lines)))
                else:
                    slice_lines = all_lines[start - 1:end]
                    numbered = '\n'.join(f'{start + i:6d}: {line}' for i, line in enumerate(slice_lines))
                    file_sections.append(f'Lines {start}-{end}:\n{numbered}')
            if file_sections:
                content = '\n\n'.join(file_sections)
                lang = 'bash' if filename.endswith('.sh') else 'python'
                sections.append(f'\n## {filename}  [{edit_note}]\n```{lang}\n{content}\n```')
        return sections

    def eval_sections(self):
        cfg = self.cfg
        sections = []
        test_cmds = cfg.get('test_cmds', [])
        if test_cmds:
            cmd_lines = [f"  - `{e['cmd']}` → label: `{e['label']}`" for e in test_cmds
                         if e.get('cmd') and e.get('label')]
            if cmd_lines:
                sections.append('\n## Evaluation Commands\nYour algorithm is evaluated by running:\n'
                                + '\n'.join(cmd_lines))
            budget_rows = []
            for e in test_cmds:
                if not e.get('cmd') or not e.get('label'):
                    continue
                compute = float(e.get('compute', 1) or 1)
                time_str = e.get('time', '1:00:00')
                if compute >= 1.0:
                    gpu_desc = f'{int(compute)} GPU(s)' if compute == int(compute) else f'{compute:.1f} GPU(s)'
                else:
                    frac = f'1/{int(round(1 / compute))}' if compute > 0 else '0'
                    gpu_desc = f'{frac} GPU'
                budget_rows.append(f"| `{e['label']}` | {gpu_desc} | {time_str} |")
            if budget_rows:
                sections.append(
                    '\n## Compute Budget\n'
                    'All evaluation runs on **NVIDIA H100 80GB** GPU(s). '
                    'Your algorithm must complete within the time limits below. '
                    'If a command exceeds its time limit, the run is killed and the result is '
                    '**invalid** (it will not count as a valid test result). '
                    'Design your model to be efficient enough to train and evaluate within these constraints.\n\n'
                    '| Command | GPUs | Time Limit |\n'
                    '| --- | --- | --- |\n'
                    + '\n'.join(budget_rows))
        return sections


# ---------------------------------------------------------------------------
# rung code -> per-region new contents
# ---------------------------------------------------------------------------
def fences_of(text):
    return [(lang, body if body.endswith('\n') else body + '\n') for lang, body in FENCE.findall(text)]


def _split_by_gaps(body, gaps):
    """A single fence that spans several editable ranges of one file carries the
    protected gap text between them; split it on those gaps (in order)."""
    parts, rest = [], body
    for g in gaps:
        i = rest.find(g)
        if i < 0:
            return None
        parts.append(rest[:i])
        rest = rest[i + len(g):]
    parts.append(rest)
    return parts


def _distinctive(line):
    t = line.strip()
    return len(t) >= 12 and any(c.isalnum() for c in t) and t not in ('return', 'pass', 'else:', 'try:')


def _trim_neighbours(new, above, below):
    """Answers often quote the protected frame around the region (the def header, the
    banner, the closing marker). Cut the fence at the nearest DISTINCTIVE protected line
    above the region that it quotes (first occurrence within the fence head) and at the
    nearest distinctive protected line below it (within the fence tail). A cut may never
    remove more than 40% of the fence — beyond that the 'frame' is really the code."""
    nl = new.splitlines(True)
    frame = {l for l in (above + below).splitlines(True) if l.strip()}

    def only_frame(chunk):          # removed lines must be frame quotes or comment/decorator lines
        return all(l in frame or l.strip().startswith(('#', '@')) for l in chunk if l.strip())
    ab = [l for l in above.splitlines(True) if _distinctive(l)]
    for anchor in reversed(ab[-6:]):
        hit = [k for k in range(min(40, len(nl) - 1)) if nl[k] == anchor]
        if hit and only_frame(nl[:hit[0] + 1]):
            nl = nl[hit[0] + 1:]
            break
    bl = [l for l in below.splitlines(True) if _distinctive(l)]
    for anchor in bl[:6]:
        hit = [k for k in range(len(nl) - 1, max(len(nl) - 40, 0), -1) if nl[k] == anchor]
        if hit and only_frame(nl[hit[0]:]):
            nl = nl[:hit[0]]
            break
    while nl and not nl[0].strip() and len(nl) > 1 and not nl[1].strip():
        nl = nl[1:]
    return ''.join(nl) if nl else new


def _base_indent(text):
    ind = [len(l) - len(l.lstrip()) for l in text.splitlines() if l.strip()]
    return min(ind) if ind else None


def _match_indent(cur, new):
    """A region that lives inside a def/class keeps its base indentation; answers quote
    the code dedented. Shift the fence so its base indent equals the region's."""
    a, b = _base_indent(cur), _base_indent(new)
    if a is None or b is None or a == b:
        return new
    delta = a - b
    out = []
    for l in new.splitlines(True):
        if not l.strip():
            out.append(l)
        elif delta > 0:
            out.append(' ' * delta + l)
        else:
            k = len(l) - len(l.lstrip())
            out.append(l[min(k, -delta):])
    return ''.join(out)


def assign_regions(sim, fences):
    """Map a rung's code fences onto the editable regions.

    Returns [(filename, region_index, new_text)] for regions that change."""
    out = []
    regs = []
    for fn in sim.editable_files():
        for ri, (s, e, txt) in enumerate(sim.regions(fn)):
            regs.append((fn, ri, s, e, txt))
    if not regs:
        raise ValueError('task has no editable region')
    if len(regs) == 1:
        fn, ri, s, e, txt = regs[0]
        body = max((b for _, b in fences), key=len)
        return [(fn, ri, body)]
    # multi-region: first try single-fence-spans-all-ranges per file
    used = set()
    by_file = {}
    for r in regs:
        by_file.setdefault(r[0], []).append(r)
    cands = list(range(len(fences)))
    for fn, rs in by_file.items():
        if len(rs) > 1:
            text = sim.read(fn).splitlines(True)
            gaps = [''.join(text[rs[i][3]:rs[i + 1][2] - 1]) for i in range(len(rs) - 1)]
            for ci in cands:
                if ci in used:
                    continue
                parts = _split_by_gaps(fences[ci][1], gaps)
                if parts and len(parts) == len(rs):
                    for (f2, ri, s, e, txt), p in zip(rs, parts):
                        out.append((f2, ri, p if p.endswith('\n') else p + '\n'))
                    used.add(ci)
                    break
    done = {(f, ri) for f, ri, _ in out}
    # remaining regions: best-ratio fence, extension-aware
    for fn, ri, s, e, txt in regs:
        if (fn, ri) in done:
            continue
        lang_want = _EXT_LANG.get(os.path.splitext(fn)[1].lower(), 'python')

        def tagged(lang):
            return lang == lang_want or (lang in ('py', 'python') and lang_want == 'python')
        best, best_r = None, 0.0
        for pass_tagged in (True, False):
            for ci, (lang, body) in enumerate(fences):
                if ci in used or (tagged(lang) != pass_tagged):
                    continue
                if not pass_tagged and lang:      # tagged with ANOTHER language: never
                    continue
                if not pass_tagged and lang_want == 'python':
                    import ast, textwrap        # an untagged fence must at least be Python
                    try:
                        ast.parse(textwrap.dedent(body))
                    except SyntaxError:
                        continue
                r = difflib.SequenceMatcher(None, txt, body, autojunk=False).quick_ratio()
                if r > best_r:
                    best, best_r = ci, r
            if best is not None and best_r >= 0.2:
                break
        if best is None or best_r < 0.2:
            continue                       # region left untouched this rung
        used.add(best)
        out.append((fn, ri, fences[best][1]))
    if not out:
        raise ValueError('no fence maps onto any editable region')
    return out


# ---------------------------------------------------------------------------
# str_replace execution through the real tool
# ---------------------------------------------------------------------------
def _aligned(state, old):
    i = state.find(old)
    j = i + len(old)
    return i >= 0 and (i == 0 or state[i - 1] == '\n') and (j == len(state) or state[j] == '\n')


def align_in_window(full, old_str, new_str, s, e):
    """harness-aligned (no trailing newline) pair, unique in the WHOLE file, widened
    only inside the editable window [s, e] (1-indexed lines)."""
    lines = full.splitlines(True)
    region = ''.join(lines[s - 1:e])
    assert region.count(old_str) == 1, 'anchor not unique inside the region'
    target_region = region.replace(old_str, new_str, 1)
    i1 = s - 1 + region.count('\n', 0, region.index(old_str))
    i2 = i1 + old_str.count('\n') + (0 if old_str.endswith('\n') else 1)
    new_lines = new_str.splitlines(True)
    for _ in range(e - s + 3):
        o = ''.join(lines[i1:i2]); n = ''.join(new_lines)
        o2 = o[:-1] if o.endswith('\n') else o
        n2 = n[:-1] if n.endswith('\n') else n
        if o2 and n2 and full.count(o2) == 1 and _aligned(full, o2):
            after = full.replace(o2, n2, 1)
            if ''.join(after.splitlines(True)[s - 1:s - 1 + len(target_region.splitlines(True))]) == target_region:
                return o2, n2
        if i1 > s - 1:
            i1 -= 1; new_lines = [lines[i1]] + new_lines
        elif i2 < e:
            new_lines = new_lines + [lines[i2]]; i2 += 1
        else:
            break
    raise ValueError('cannot express the edit as a unique in-window str_replace')


def apply_region_ops(sim, fn, ri, new_text, plan_ops, max_ops=None):
    """Turn region ri of fn into new_text via the real tool; returns the executed
    [(old_str, new_str, result)] list."""
    s, e, cur = sim.regions(fn)[ri]
    lines = sim.read(fn).splitlines(True)
    new_text = _trim_neighbours(new_text, ''.join(lines[max(0, s - 9):s - 1]), ''.join(lines[e:e + 8]))
    new_text = _match_indent(cur, new_text)
    if cur == new_text:
        return []
    pairs = plan_ops(cur, new_text)
    if max_ops is not None and len(pairs) > max_ops:
        pairs = [(cur, new_text)]
    target = new_text

    import copy as _copy

    def run(pairs):
        done = []
        path = sim.tools._resolve_workspace_path(fn)
        snap = (path.read_text(), _copy.deepcopy(sim.tools.live_protected_ranges),
                list(sim.tools._history))
        for old, new in pairs:
            s, e, cur = sim.regions(fn)[ri]
            full = sim.read(fn)
            o2, n2 = align_in_window(full, old, new, s, e)
            res = sim.edit(fn, o2, n2)
            if not res.startswith('OK: Replaced 1 occurrence'):
                # roll the partial chain back deterministically (file, ranges, history)
                path.write_text(snap[0]); sim.tools.live_protected_ranges = snap[1]
                sim.tools._history[:] = snap[2]
                raise ValueError('harness rejected the edit: ' + res[:300].replace('\n', ' | '))
            if '[matched after:' in res.split('\n', 1)[0]:
                raise ValueError('harness matched loosely: ' + res[:200])
            done.append((o2, n2, res))
        return done

    try:
        done = run(pairs)
    except ValueError as err:
        if len(pairs) == 1 or 'rejected' not in str(err):
            raise
        # a chain whose intermediate state does not parse: the harness's syntax gate
        # would reject it live, so replace the region in one op instead
        s, e, cur = sim.regions(fn)[ri]
        done = run([(cur, target)])
    s, e, cur = sim.regions(fn)[ri]
    if cur != target:
        raise ValueError('region content after edits != rung code')
    return done


# ---------------------------------------------------------------------------
# Lint: replay a filled/skeleton episode through a FRESH real workspace and demand
# byte-equality of every tool result and of the harness-derived prompt sections.
# ---------------------------------------------------------------------------
def _env_pythons(cfg):
    out = []
    pkgs = {e.get('package') for e in cfg.get('test_cmds', []) if e.get('package')}
    for pk in sorted(pkgs):
        for d in Path(os.path.expanduser('~/miniconda3/envs')).glob('mlsbench-*'):
            if _norm(d.name[len('mlsbench-'):]) == _norm(pk) and (d / 'bin' / 'python').exists():
                out.append(str(d / 'bin' / 'python'))
    return out


def replay_check(d, _in_env=False):
    task = d['task']
    problems = []
    n_tests = d['n_rungs']
    try:
        sim = Sim(task, n_tests)
    except ModuleNotFoundError as e:
        if _in_env:
            return ['replay needs a module even the package env lacks: %s' % e]
        # the harness loads the task parser, which needs the task package's conda env
        import subprocess
        cfg = json.load(open(f'{ROOT}/tasks/{task}/config.json'))
        for py in _env_pythons(cfg):
            tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, prefix='agv2rep_')
            json.dump(d, tmp); tmp.close()
            r = subprocess.run([py, os.path.abspath(__file__), '--replay', tmp.name], capture_output=True,
                               text=True, timeout=1200, cwd=os.getcwd())
            os.unlink(tmp.name)
            if r.returncode == 0 and '@@PROBLEMS@@' in r.stdout:
                return json.loads(r.stdout.rsplit('@@PROBLEMS@@', 1)[1])
        return ['replay could not run in any interpreter: %s' % e]
    try:
        prompt = d['messages'][0]['content']
        for sec in sim.file_sections() + sim.eval_sections():
            if sec not in prompt:
                problems.append('prompt lacks harness section: ' + sec[:60].replace('\n', ' '))
        msgs = d['messages']
        for i, m in enumerate(msgs):
            if m['role'] != 'assistant':
                continue
            fn = m['tool_calls'][0]['function']
            if fn['name'] != 'edit':
                continue
            a = fn['arguments']
            if a['op'] != 'str_replace':
                problems.append('msg %d: op %s not allowed in mls-real mode' % (i, a['op']))
                continue
            res = sim.edit(a['filename'], a['old_str'], a['new_str'])
            rec = msgs[i + 1]['content'] if i + 1 < len(msgs) else None
            if res != rec:
                problems.append('msg %d: harness result differs from recorded (%s...)' % (i, res[:80].replace('\n', ' ')))
    finally:
        sim.close()
    return problems


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--replay':
        _d = json.load(open(sys.argv[2], encoding='utf-8'))
        _p = replay_check(_d, _in_env=True)
        sys.stdout.write('\n@@PROBLEMS@@' + json.dumps(_p))
