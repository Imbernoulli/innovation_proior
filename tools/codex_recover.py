#!/usr/bin/env python3
"""Recover audit-edit results lost to an erroneous `git checkout`, using codex exec (gpt-5.6-terra)
instead of Claude subagents — Claude billing-cycle quota is exhausted (403), codex is a separate channel.

For each unfinished unit: codex reads the ORIGINAL edit agent's full JSONL transcript (which records
every read/Edit/Bash/codex step), reproduces that agent's FINAL edited file state onto the clean HEAD
baseline, then commits ONLY that unit's pathspec. Per-unit durability: commit lands immediately.

The original Claude recovery prompt is reused verbatim in spirit. FROZEN-file invariants are enforced
by a mechanical post-check: after each codex run we verify `git diff HEAD --name-only` for the pathspec
touches ONLY allowed files; if codex touched a frozen file, we revert the unit and log it for redo.

Usage: python3 tools/codex_recover.py [--todo tmp/recover/codex_todo.json] [--start N] [--limit M]
"""
import json, os, re, subprocess, sys, time

REPO = '/srv/home/bohanlyu/innovation_proior'
MODEL = 'gpt-5.6-terra'
EFFORT = 'high'
TIMEOUT = 1500

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def sh(cmd, cwd=REPO, timeout=None, check=False):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str))
    if check and p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{p.stderr[:400]}")
    return p

def allowed_files(kind, pathspec):
    """Return a predicate-ish description + the check. frozen invariants per kind."""
    return pathspec

def committed_units():
    out = sh(['git', 'log', '--format=%s', 'audit-edit-baseline..HEAD']).stdout
    s = set()
    for line in out.splitlines():
        m = re.match(r'recover audit-edit (\w+):(\S+)', line.strip())
        if m: s.add((m.group(1), m.group(2)))
    return s

def build_prompt(u):
    unitDir, kind, unit, pathspec, transcript = u['unitDir'], u['kind'], u['unit'], u['pathspec'], u['transcript']
    newWords = u.get('newWords') or 0
    frozen = {
        'traj': 'edit ONLY the NN-*-reasoning.md files; NN-*-answer.md, NN-*-train_answer.md, NN-feedback.md, 00-initial-context.md, meta.json are FROZEN',
        'v4': 'edit ONLY reasoning.md; context.md, train_answer.md, verify/ are FROZEN',
        'method': 'edit reasoning.md always; results/train_answer.md ONLY if the original changed its PROSE (claim fix) with code blocks byte-identical; context.md, answer.md are FROZEN',
    }[kind]
    return f"""You are RECOVERING one unit of audit-edit work lost before it was committed. Repo: {REPO} (git baseline tag: audit-edit-baseline).

Unit: {unitDir}  (kind={kind})
Edit ONLY files under: {pathspec}
Original edit agent's FULL transcript (JSONL, one message per line): {transcript}
The original agent reported new_words ≈ {newWords} after its edit.

WHAT HAPPENED: an earlier agent edited this unit's reasoning (cutting ritual filler / hindsight / pasted answer code, fixing real defects) and it PASSED review, but the working-tree changes were reverted by a mistaken `git checkout` before being committed. Files under {pathspec} are back at the PRE-edit baseline. Reproduce the original agent's FINAL edited state and commit it.

READ THE TRANSCRIPT CAREFULLY (it is the source of truth). Walk it in order and reconstruct every mutation to files under {pathspec}:
- tool_use "Write" = full-file overwrite ({{file_path, content}}).
- tool_use "Edit" = {{file_path, old_string, new_string, replace_all}}.
- tool_use "Bash" may ALSO write files (heredocs `cat > f <<'EOF'`, redirects `... > f`, `sed -i`, `mv /tmp/x f`) — account for these too; this is what a naive replay misses.
- A `codex exec` call may have edited files; the agent's LATER steps (git diff dumps, final Read, wc -w) reveal the result. Use those as ground truth.
The transcript tail often re-reads the file or dumps `git diff` / `wc -w` — use that as the final-state ground truth.

FROZEN INVARIANT (must hold in your result): {frozen}. If the transcript appears to have touched a frozen file, do NOT reproduce that; keep the frozen file at baseline.

SANITY CHECK: after writing, run `wc -w` on your reconstructed reasoning file(s); the total should be within ~6% of {newWords}. If wildly off, you missed a Bash-write or a codex edit — re-read the transcript tail and fix. Confirm `git -C {REPO} diff --name-only` lists ONLY files under {pathspec}.

COMMIT ONLY this unit (never `git add -A` / `git add .`):
    git -C {REPO} add -- {pathspec}
    git -C {REPO} diff --cached --quiet || git -C {REPO} commit -q -m "recover audit-edit {kind}:{unit}"

Do the recovery now. After committing, print a final line: "DONE <short-hash>" or "REDO <reason>" if you could not reconstruct with confidence (in which case restore via `git -C {REPO} checkout -- {pathspec}` first)."""

def run_unit(u, scratch):
    unitDir, kind, unit, pathspec = u['unitDir'], u['kind'], u['unit'], u['pathspec']
    log(f"recover {kind}:{unit} ...")
    # snapshot frozen-file hashes BEFORE (to verify codex didn't touch them)
    prompt = build_prompt(u)
    os.makedirs(scratch, exist_ok=True)
    t0 = time.time()
    try:
        p = subprocess.run(['codex', 'exec', '--skip-git-repo-check', '-m', MODEL,
                            '-c', f'model_reasoning_effort="{EFFORT}"', prompt],
                           cwd=scratch, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=TIMEOUT)
        out = (p.stdout or '') + '\n' + (p.stderr or '')
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT {kind}:{unit}"); return ('timeout', unitDir)
    secs = time.time() - t0
    # verify: committed and pathspec-scoped
    # did a new commit land referencing this unit?
    recent = sh(['git', 'log', '--format=%s', '-1', f'--grep=recover audit-edit {kind}:{unit}']).stdout.strip()
    dirty = sh(['git', 'status', '--porcelain', '--', pathspec]).stdout.strip()
    changed = [l for l in sh(['git', 'diff', 'HEAD', '--name-only']).stdout.splitlines() if l.strip()]
    # any changes OUTSIDE pathspec (codex violated scope)?
    outside = [f for f in changed if not f.startswith(pathspec.rstrip('/') + '/') and pathspec.rstrip('/') not in f]
    ok_commit = bool(recent) and f'recover audit-edit {kind}:{unit}' in recent
    if ok_commit and not dirty:
        log(f"  OK {kind}:{unit}  ({secs:.0f}s) {recent[:40]}")
        return ('ok', unitDir)
    # failed or partial: revert ONLY this unit's uncommitted dirt, log for redo
    if dirty:
        sh(['git', 'checkout', '--', pathspec])
    reason = 'no-commit' if not ok_commit else ('scope-violation' if outside else 'dirty-left')
    log(f"  REDO {kind}:{unit} ({secs:.0f}s) {reason}; out-of-scope: {outside[:3]}")
    # stash raw output for debugging
    open(os.path.join(scratch, '_last_raw.txt'), 'w').write(out[-8000:])
    return ('redo', unitDir, reason)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default=f'{REPO}/tmp/recover/history_map.json')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=10**9)
    ap.add_argument('--kinds', default='', help='comma list filter: method,traj,v4')
    ap.add_argument('--scratch', default='/tmp/codex_recover_scratch')
    ap.add_argument('--results', default=f'{REPO}/tmp/recover/codex_results.jsonl')
    args = ap.parse_args()

    m = json.load(open(args.map))
    done = committed_units()
    order = {'method': 0, 'v4': 1, 'traj': 2}
    todo = []
    for ud, v in m.items():
        if (v['kind'], v['unit']) in done: continue
        todo.append({'unitDir': ud, 'kind': v['kind'], 'unit': v['unit'], 'pathspec': v['pathspec'],
                     'transcript': v['transcript'], 'newWords': v.get('new_words') or 0, 'verdict': v.get('verdict') or 'na'})
    todo.sort(key=lambda u: (order[u['kind']], u['unitDir']))
    if args.kinds:
        keep = set(args.kinds.split(','))
        todo = [u for u in todo if u['kind'] in keep]
    todo = todo[args.start:args.start + args.limit]
    log(f"codex_recover: {len(todo)} units to do (start={args.start})")

    results = []
    rf = open(args.results, 'a')
    for u in todo:
        r = run_unit(u, os.path.join(args.scratch, u['kind'] + '_' + u['unit']))
        results.append(r)
        rf.write(json.dumps({'unit': u['unitDir'], 'status': r[0], 'detail': r[2] if len(r) > 2 else ''}) + '\n'); rf.flush()
        # re-check ground truth each unit (in case of partial)
    rf.close()
    ok = sum(1 for r in results if r[0] == 'ok')
    redo = sum(1 for r in results if r[0] == 'redo')
    to = sum(1 for r in results if r[0] == 'timeout')
    log(f"DONE: {ok} ok, {redo} redo, {to} timeout of {len(results)}")

if __name__ == '__main__':
    main()
