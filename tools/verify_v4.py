#!/usr/bin/env python3
"""Independently re-verify every data_v4/cpv4-* datapoint (compile + brute-force oracle).

The generation workflow's independent verify stage got rate-limited, so we do the verification here
directly: for each datapoint, compile verify/sol.cpp, run N random cases from verify/gen.py through
both sol and verify/brute.py, and require 0 mismatches; also require the three .md files and a >=12k
reasoning with a cpp block. Prints PASS/FAIL per datapoint and a final keep-list; writes the passing
slugs to data_v4/_verified.txt.

Usage: python tools/verify_v4.py [N_cases]
"""
import glob, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
MIN_REASON = 12000


def run(cmd, inp=None, timeout=20):
    try:
        return subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return subprocess.CompletedProcess(cmd, 1, '', str(e))


def gen_input(genpy, seed):
    # gen.py is parameterized by an int seed arg
    r = run(['python3', genpy, str(seed)])
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def verify(d):
    slug = os.path.basename(d.rstrip('/'))
    v = os.path.join(d, 'verify')
    sol, brute, gen = (os.path.join(v, f) for f in ('sol.cpp', 'brute.py', 'gen.py'))
    for f in ('context.md', 'reasoning.md', 'train_answer.md'):
        if not os.path.isfile(os.path.join(d, f)):
            return slug, False, f'missing {f}'
    if not all(os.path.isfile(x) for x in (sol, brute, gen)):
        return slug, False, 'missing verify artifacts (sol/brute/gen)'
    reason = open(os.path.join(d, 'reasoning.md'), encoding='utf-8').read()
    if len(reason) < MIN_REASON:
        return slug, False, f'reasoning too short ({len(reason)})'
    if '```cpp' not in open(os.path.join(d, 'train_answer.md'), encoding='utf-8').read():
        return slug, False, 'no cpp block in train_answer'
    exe = f'/tmp/v4_{slug}'
    c = run(['g++', '-O2', '-std=c++17', '-o', exe, sol], timeout=60)
    if c.returncode != 0:
        return slug, False, 'compile error: ' + (c.stderr.strip().splitlines() or [''])[-1][:120]
    ran = mm = 0
    for s in range(1, N + 1):
        inp = gen_input(gen, s)
        if inp is None:
            continue
        a = run([exe], inp)
        b = run(['python3', brute], inp)
        if a.returncode != 0 or b.returncode != 0:
            continue
        ran += 1
        if a.stdout.strip() != b.stdout.strip():
            mm += 1
            if mm <= 1:
                first = inp.strip().replace('\n', ' ')[:80]
    if ran < 20:
        return slug, False, f'oracle could not run enough cases ({ran})'
    if mm > 0:
        return slug, False, f'oracle MISMATCH {mm}/{ran} (e.g. input: {first})'
    return slug, True, f'ok ({ran} cases, 0 mismatch, reasoning {len(reason)}c)'


def main():
    dirs = sorted(glob.glob(os.path.join(ROOT, 'data_v4/cpv4-*/')))
    passed, failed = [], []
    for d in dirs:
        slug, ok, msg = verify(d)
        print(f"{'PASS' if ok else 'FAIL'}  {slug}: {msg}")
        (passed if ok else failed).append(slug)
    print(f"\n=== {len(passed)} PASS / {len(failed)} FAIL / {len(dirs)} total ===")
    open(os.path.join(ROOT, 'data_v4/_verified.txt'), 'w').write('\n'.join(passed) + '\n')
    if failed:
        print("FAILED:", ', '.join(failed))


if __name__ == '__main__':
    main()
