#!/usr/bin/env python3
"""Filter the capability-maintenance set by GENERATION (Qwen3.6 vs Qwen3.8) + difficulty.

Policy (user, 2026-08-28):
  * Qwen3.6 rows: keep only the hard ones -- drop pass_rate == 1.0.
  * Qwen3.8 rows: keep ALL, including pass_rate == 1.0.
  * Drop every row whose first assistant turn opens with "Here's a thinking process",
    a Qwen3.6 distillation artefact.

Recovering the generation, since no shipped file carries a model field
--------------------------------------------------------------------
sft/innovation_wave3_sft.jsonl.gz has only conversations/pass_rate/samples_used/system.
`source: Qwen3.8-27B` lives in traces/<domain>.q38.jsonl, which is not on this machine.
So generation is recovered from GIT HISTORY, the same method bl3615 used in
/scratch/gpfs/CHIJ/bohan/fs/tinker_line/wave3_q38_indices.json
("method": "sha256(conversations) set-diff vs git 136f69979"):

    136f69979  2026-08-13  wave-3: 2,220              <- last commit before Qwen3.8
    a84efca35  2026-08-15  2,254 - first Qwen3.8-27B teacher keepers (25)
    8a145ab17  2026-08-19  wave-3 FINAL: 5,291

Rows whose conversations hash is present at 136f69979 are Qwen3.6; everything added
since is Qwen3.8. That gives q36 2,220 / q38 3,071 against the README's stated 3,062
q38 keepers -- 9 apart, i.e. dedup noise.

Two independent signals confirm the labelling rather than assuming it:
  * "Here's a thinking process" openers: 40.5% of q36 vs 0.2% of q38.
  * pass_rate == 1.0: 41.4% of q36 vs 63.0% of q38 -- which is exactly why the
    generation split matters. Dropping pass_rate == 1.0 wholesale would have deleted
    1,934 Qwen3.8 rows that this policy keeps.

Maintenance rows that are NOT in wave3 come from wave-2, whose raw keepers record
`source: Qwen3.6-27B` for 715 of 741 and which is already filtered to round-0
accuracy <= 0.5. They are kept (minus the thinking-process ones): already hard, and no
pass_rate to test.

Usage:
  python sft/filter_maintain.py --in  $D/data/maintain_hard.jsonl \
                                --out $D/data/maintain_hard_f.jsonl
"""
from __future__ import annotations
import argparse, collections, gzip, hashlib, io, json, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAVE3 = 'sft/innovation_wave3_sft.jsonl.gz'
PRE_Q38_REV = '136f69979'          # 2026-08-13, last wave-3 commit before Qwen3.8
THINKING = re.compile(r"^\s*Here'?s\s+a\s+thinking\s+process", re.I)
_THINK_OPEN = re.compile(r'^\s*<think>\s*')


def conv_hash(row: dict) -> str:
    return hashlib.sha256(
        json.dumps(row['conversations'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def user_key(row: dict) -> str | None:
    for m in row.get('conversations') or []:
        if m.get('from') in ('human', 'user'):
            return hashlib.sha1((m.get('value') or '').strip().encode()).hexdigest()
    return None


def first_assistant(row: dict) -> str:
    for m in row.get('conversations') or []:
        if m.get('from') == 'gpt':
            return _THINK_OPEN.sub('', m.get('value') or '')
    return ''


def build_index() -> dict[str, tuple[str, float | None]]:
    """user_key -> (generation, pass_rate) for every wave-3 row."""
    blob = subprocess.run(['git', '-C', REPO, 'show', f'{PRE_Q38_REV}:{WAVE3}'],
                          capture_output=True).stdout
    if not blob:
        sys.exit(f'cannot read {WAVE3} at {PRE_Q38_REV}; generation labelling needs git history')
    pre = {conv_hash(json.loads(l)) for l in gzip.open(io.BytesIO(blob), 'rt')}
    idx = {}
    with gzip.open(os.path.join(REPO, WAVE3), 'rt') as fh:
        for line in fh:
            d = json.loads(line)
            k = user_key(d)
            if k is not None:
                idx[k] = ('q36' if conv_hash(d) in pre else 'q38', d.get('pass_rate'))
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', dest='dst', required=True)
    ap.add_argument('--keep-thinking', dest='drop_thinking', action='store_false', default=True)
    args = ap.parse_args()

    idx = build_index()
    n = 0
    stats = collections.Counter()
    with open(args.dst, 'w', encoding='utf-8') as out:
        for line in open(args.src, encoding='utf-8'):
            row = json.loads(line)
            n += 1
            gen, rate = idx.get(user_key(row), ('wave2', None))
            if args.drop_thinking and THINKING.match(first_assistant(row)):
                stats[f'DROP {gen} thinking-process'] += 1
                continue
            if gen == 'q36' and rate == 1.0:
                stats['DROP q36 pass_rate==1'] += 1
                continue
            stats[f'keep {gen}'] += 1
            out.write(json.dumps(row, ensure_ascii=False) + '\n')

    kept = sum(v for k, v in stats.items() if k.startswith('keep'))
    print(f'{args.src} -> {args.dst}')
    for k in sorted(stats):
        print(f'  {k:32s} {stats[k]:5d}')
    print(f'  {"KEPT":32s} {kept:5d} / {n} ({100 * kept / n:.1f}%)')


if __name__ == '__main__':
    main()
