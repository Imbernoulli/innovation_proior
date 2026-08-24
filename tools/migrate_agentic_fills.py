#!/usr/bin/env python3
"""Re-key existing agentic_v2_fills.json files onto a regenerated skeleton.

Used for the 2026-08-24 contract migration (str_replace-only -> live rewrite
contract): slot seq numbers shifted and rungs that collapsed into a single
op='rewrite' lost their followup slots. Fills are matched structurally —
(kind, rung, slug) with order-preserving ordinal for followups — so the prose
survives; surplus followup fills (their hunks merged into the rewrite) are
dropped and reported. After re-keying, the applier runs (apply + lint).

Usage: python3 tools/migrate_agentic_fills.py [TASK ...]   (default: all with fills)
"""
import glob
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

SENT = re.compile(r'\[\[([^\]]+)\]\]')
KEY = re.compile(r'^(THINK|SAY)\s+(.*?)\s*seq=\d+$')


def norm(key):
    """Strip the seq= tail: 'THINK kind=design rung=1 slug=gin seq=3' -> stable part."""
    m = KEY.match(key.strip())
    if not m:
        return None
    return (m.group(1) + ' ' + m.group(2)).strip()


def slots_of(skel):
    out = []
    for msg in skel['messages']:
        if msg['role'] != 'assistant':
            continue
        for field in ('reasoning_content', 'content'):
            v = (msg.get(field) or '').strip()
            ms = SENT.fullmatch(v)
            if ms:
                out.append(ms.group(1))
    return out


def migrate(task):
    d = f'trajectories/{task}'
    fp = f'{d}/agentic_v2_fills.json'
    skel = json.load(open(f'{d}/agentic_v2.json', encoding='utf-8'))
    fills = json.load(open(fp, encoding='utf-8'))['fills']

    # bucket old fills by normalized key, order-preserving (seq order = file order
    # is not guaranteed, so sort by the numeric seq in the key)
    def seq_of(k):
        m = re.search(r'seq=(\d+)', k)
        return int(m.group(1)) if m else 0
    old_by_norm = {}
    for k in sorted(fills, key=seq_of):
        n = norm(k)
        if n is None:
            print(f'{task}: skipping unparseable fill key {k!r}')
            continue
        old_by_norm.setdefault(n, []).append(fills[k])

    new_fills, missing, used = {}, [], {}
    for slot in slots_of(skel):
        n = norm(slot)
        pool = old_by_norm.get(n, [])
        idx = used.get(n, 0)
        if idx < len(pool):
            new_fills[slot] = pool[idx]
            used[n] = idx + 1
        else:
            missing.append(slot)
    dropped = sum(len(v) - used.get(k, 0) for k, v in old_by_norm.items())

    if missing:
        print(f'{task}: {len(missing)} slot(s) have NO source fill: {missing[:4]}')
        return False
    json.dump({'task': task, 'fills': new_fills}, open(fp, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    r = subprocess.run([sys.executable, 'tools/apply_agentic_fills.py', task],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    print(f'{task}: {len(new_fills)} slots re-keyed, {dropped} surplus followup fill(s) '
          f'dropped, apply {"clean" if ok else "FAILED"}')
    if not ok:
        print(r.stdout.strip())
    return ok


def main():
    tasks = sys.argv[1:] or sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob('trajectories/*/agentic_v2_fills.json'))
    bad = [t for t in tasks if not migrate(t)]
    print(f'\nmigrated {len(tasks) - len(bad)}/{len(tasks)}; failed: {bad}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
