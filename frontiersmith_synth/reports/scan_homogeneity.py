#!/usr/bin/env python3
"""scan_homogeneity.py -- cross-problem structural-diversity gate.

Motivation: the 2026-07-08 "wave 2" incident, where 500 problems passed every
per-problem gate yet were ONE code template instantiated 500 times (only an
integer salt + the theme string differed). Per-problem validation cannot see
this; this scan compares problems AGAINST EACH OTHER.

Two signals per problem:
  * skeleton hash  -- concatenated source of gen/checker/evaluator/solution files
                      with digits stripped and whitespace collapsed, md5'd.
                      Clones that differ only in seeds/constants collide.
  * statement shape -- statement text with digits stripped, theme-ish capitalized
                      words masked, whitespace collapsed, md5'd. Template
                      statements that differ only in the story skin collide.

Gate: FAIL if any skeleton-hash group or statement-shape group exceeds
--max-clones (default 3) members.

Usage:
  python3 reports/scan_homogeneity.py                 # scan problems/, gate at 3
  python3 reports/scan_homogeneity.py --ids id1 id2   # scan a subset
  python3 reports/scan_homogeneity.py --max-clones 1  # strictest
Exit 0 = PASS, 1 = FAIL (prints offending groups).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

SYNTH = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "gen.py", "gen.cpp", "verify.py", "chk.cc", "evaluator.py", "counter.py",
    "solutions/trivial.py", "solutions/trivial.cpp",
    "solutions/strong.py", "solutions/strong.cpp",
)


def _norm_code(text: str) -> str:
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _norm_statement(text: str) -> str:
    text = re.sub(r"\d+", "#", text)
    # mask capitalized theme-ish words so "Aquarium Habitat" == "School Menu"
    text = re.sub(r"\b[A-Z][a-zA-Z]*\b", "W", text)
    # mask inline/bold code-ish tokens that usually carry the theme
    text = re.sub(r"\*\*[^*]+\*\*", "T", text)
    text = re.sub(r"`[^`]+`", "C", text)
    text = re.sub(r"\s+", " ", text)
    return text


def skeleton_hash(pdir: Path) -> str:
    parts = []
    for rel in CODE_FILES:
        f = pdir / rel
        if f.exists():
            parts.append(rel + "\x00" + _norm_code(f.read_text(errors="ignore")))
    return hashlib.md5("\x01".join(parts).encode()).hexdigest()[:12]


def statement_hash(pdir: Path) -> str:
    for name in ("statement.md", "statement.txt"):
        f = pdir / name
        if f.exists():
            return hashlib.md5(_norm_statement(f.read_text(errors="ignore")).encode()).hexdigest()[:12]
    return "no-statement"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-dir", type=Path, default=SYNTH / "problems")
    ap.add_argument("--ids", nargs="*", help="only scan these problem ids")
    ap.add_argument("--max-clones", type=int, default=3,
                    help="max problems allowed to share one skeleton/statement shape")
    ap.add_argument("--json", action="store_true", help="emit machine-readable report")
    args = ap.parse_args()

    dirs = sorted(d for d in args.problems_dir.iterdir() if d.is_dir())
    if args.ids:
        want = set(args.ids)
        dirs = [d for d in dirs if d.name in want]
    if not dirs:
        print("no problems to scan", file=sys.stderr)
        return 1

    skel = collections.defaultdict(list)
    stmt = collections.defaultdict(list)
    for d in dirs:
        skel[skeleton_hash(d)].append(d.name)
        stmt[statement_hash(d)].append(d.name)

    bad = []
    for kind, groups in (("skeleton", skel), ("statement", stmt)):
        for h, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(members) > args.max_clones:
                bad.append({"kind": kind, "hash": h, "count": len(members),
                            "members": members[:10]})

    report = {
        "scanned": len(dirs),
        "distinct_skeletons": len(skel),
        "distinct_statement_shapes": len(stmt),
        "max_clones_allowed": args.max_clones,
        "violations": bad,
        "verdict": "FAIL" if bad else "PASS",
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"scanned={report['scanned']}  skeletons={report['distinct_skeletons']}  "
              f"statement_shapes={report['distinct_statement_shapes']}  -> {report['verdict']}")
        for v in bad:
            print(f"  VIOLATION [{v['kind']}] {v['count']} problems share {v['hash']}: "
                  f"{', '.join(v['members'][:6])}{' ...' if v['count'] > 6 else ''}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
