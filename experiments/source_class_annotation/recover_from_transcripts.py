#!/usr/bin/env python3
"""Recover methods/<slug>/{notes,refs,src,code}/* file contents from Claude Code and Codex
transcripts: every full-file read (Read tool without offset/limit, `cat path`, `cat -n path`)
whose result was not truncated is a verbatim copy of the file at that time. Keeps the latest
full read per path; writes to a staging dir; never touches the repo."""
import json, re, os, glob, sys, time
STAGE = '/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/cbc9e24b-cb14-40d1-996b-d3b8209e16b4/scratchpad/recovered'
REPO = '/srv/home/bohanlyu/innovation_proior'
PATH_RE = re.compile(r'(?:^|/)methods/([A-Za-z0-9._-]+)/(notes|refs|src|code)/([^\s"\'`|;&<>]+)$')
CAT_RE = re.compile(r'^\s*cat\s+(?:-n\s+)?(?:-A\s+)?["\']?([^\s"\'|;&<>]+)["\']?\s*$')
TRUNC_MARKERS = ('Output too large', '[truncated', '(truncated', '... [', 'lines truncated', 'output truncated')
best = {}   # relpath -> (full:bool, ts, chars, text, src)
stats = {'files':0,'reads':0,'full':0,'partial':0}

def norm(p):
    p = p.strip()
    if p.startswith(REPO + '/'): p = p[len(REPO)+1:]
    if p.startswith('./'): p = p[2:]
    m = PATH_RE.search(p)
    if not m: return None
    return f"methods/{m.group(1)}/{m.group(2)}/{m.group(3)}"

def strip_numbers(txt, sep):
    out = []
    for line in txt.split('\n'):
        m = re.match(r'^\s*\d+' + sep, line)
        out.append(line[m.end():] if m else line)
    return '\n'.join(out)

def consider(rel, text, full, ts, src):
    stats['reads'] += 1
    if any(k in text[-400:] for k in TRUNC_MARKERS) or any(k in text[:300] for k in TRUNC_MARKERS):
        full = False
    stats['full' if full else 'partial'] += 1
    cur = best.get(rel)
    key = (full, ts if full else len(text))
    if cur is None or (key[0], key[1]) > (cur[0], cur[1] if cur[0] else cur[2]):
        best[rel] = (full, ts, len(text), text, src)

def text_of(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        return '\n'.join(x.get('text','') for x in content if isinstance(x, dict) and x.get('type') == 'text')
    return ''

def scan_claude(path):
    pending = {}
    try:
        with open(path, errors='replace') as fh:
            for line in fh:
                if '"tool_use"' not in line and '"tool_result"' not in line: continue
                try: d = json.loads(line)
                except Exception: continue
                ts = d.get('timestamp', '')
                msg = d.get('message', {}); content = msg.get('content') if isinstance(msg, dict) else None
                if not isinstance(content, list): continue
                for b in content:
                    if not isinstance(b, dict): continue
                    if b.get('type') == 'tool_use':
                        pending[b.get('id')] = (b.get('name'), b.get('input', {}), ts)
                    elif b.get('type') == 'tool_result':
                        tu = pending.pop(b.get('tool_use_id'), None)
                        if not tu: continue
                        name, inp, ts0 = tu
                        txt = text_of(b.get('content'))
                        if not txt: continue
                        if name == 'Read':
                            rel = norm(str(inp.get('file_path', '')))
                            if not rel: continue
                            full = not inp.get('offset') and not inp.get('limit')
                            consider(rel, strip_numbers(txt, r'\t'), full, ts0, path)
                        elif name == 'Bash':
                            cmd = str(inp.get('command', ''))
                            m = CAT_RE.match(cmd)
                            if m:
                                rel = norm(m.group(1))
                                if not rel: continue
                                body = strip_numbers(txt, r'\t') if re.match(r'^\s*cat\s+-n', cmd) else txt
                                consider(rel, body, True, ts0, path)
    except Exception as e:
        print('ERR', path, e, file=sys.stderr)

def scan_codex(path):
    pending = {}
    try:
        with open(path, errors='replace') as fh:
            for line in fh:
                if 'function_call' not in line: continue
                try: d = json.loads(line)
                except Exception: continue
                p = d.get('payload') or {}
                t = p.get('type')
                ts = d.get('timestamp', '')
                if t == 'function_call':
                    try: args = json.loads(p.get('arguments') or '{}')
                    except Exception: args = {}
                    cmd = args.get('command') or args.get('cmd') or ''
                    if isinstance(cmd, list): cmd = ' '.join(cmd[-1:]) if len(cmd) >= 3 and cmd[0] in ('bash','sh','zsh') else ' '.join(cmd)
                    pending[p.get('call_id')] = (str(cmd), ts)
                elif t == 'function_call_output':
                    tu = pending.pop(p.get('call_id'), None)
                    if not tu: continue
                    cmd, ts0 = tu
                    out = p.get('output')
                    if isinstance(out, dict): out = out.get('output') or out.get('content') or ''
                    out = str(out)
                    out = re.sub(r'^(?:Exit code: \d+\n)?(?:Wall time: [^\n]*\n)?(?:Output:\n)?', '', out)
                    m = CAT_RE.match(cmd)
                    if m:
                        rel = norm(m.group(1))
                        if rel: consider(rel, strip_numbers(out, r'\t') if re.match(r'^\s*cat\s+-n', cmd) else out, True, ts0, path)
    except Exception as e:
        print('ERR', path, e, file=sys.stderr)

t0 = time.time()
files = sorted(glob.glob(os.path.expanduser('~/.claude/projects/*/*.jsonl')) + glob.glob(os.path.expanduser('~/.claude/projects/*/*/subagents/**/agent-*.jsonl'), recursive=True))
for i, f in enumerate(files):
    scan_claude(f); stats['files'] += 1
    if i % 500 == 0: print(f'claude {i}/{len(files)} reads={stats["reads"]} paths={len(best)} {time.time()-t0:.0f}s', flush=True)
cfiles = sorted(glob.glob(os.path.expanduser('~/.codex/sessions/2026/*/*/*.jsonl')))
for i, f in enumerate(cfiles):
    scan_codex(f); stats['files'] += 1
    if i % 500 == 0: print(f'codex {i}/{len(cfiles)} reads={stats["reads"]} paths={len(best)} {time.time()-t0:.0f}s', flush=True)
manifest = {}
n_written = 0
for rel, (full, ts, chars, text, src) in best.items():
    manifest[rel] = {'full': full, 'ts': ts, 'chars': chars, 'src': os.path.basename(src)}
    if full:
        out = os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh: fh.write(text)
        n_written += 1
json.dump(manifest, open(os.path.join(STAGE, 'manifest.json'), 'w'), indent=0)
print('DONE', stats, 'paths', len(best), 'written', n_written, f'{time.time()-t0:.0f}s')
