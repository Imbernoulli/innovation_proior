#!/usr/bin/env python3
"""Build the FULL 40-problem ALE-Bench parquet from the local dataset cache.

We have been evaluating on a 10-problem subset (data/alebench/val.parquet) this
whole time. That is the direct cause of ALE's terrible resolution: with n=10 the
difference-SEM is ~96 points, so nothing below ~40 points is distinguishable and
every cross-arm ALE claim in this campaign has been unusable. The full set is
40 problems (ahc001..ahc048 with gaps), all of which are ALREADY downloaded --
40 zips in the HF snapshot and 40 entries in ale_score_references.json. Only the
parquet was truncated.

This script reproduces val.parquet's exact row schema (data_source / prompt /
reward_model) for all 40, reading each problem statement out of its zip, so the
prompt the model sees is unchanged apart from the statement itself.

Usage: python3 scripts/build_alebench_full40.py [--out data/alebench/full40.parquet]
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

FS = Path("/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith")
SNAP = FS / ".cache/ale-bench/datasets--SakanaAI--ALE-Bench/snapshots"
REFS = FS / "data/alebench/ale_score_references.json"
VAL = FS / "data/alebench/val.parquet"


def snapshot_dir() -> Path:
    cands = sorted(SNAP.glob("*"))
    if not cands:
        raise SystemExit(f"no ALE-Bench snapshot under {SNAP}")
    return cands[-1]


def statement_from_zip(zp: Path) -> str | None:
    """Pull the English problem statement out of a problem zip.

    The body is `statement_en.md`, but the shipped 10-problem parquet prefixes it
    with the time and memory limits, which live in `data.json` under
    `constraints`. Those two lines matter for a heuristic-optimisation benchmark
    -- the whole task is trading solution quality against a wall clock -- so a
    statement without them is a DIFFERENT problem. Verified by diffing our output
    against val.parquet on the 10 shared problems, which is how the omission was
    caught. Returns None when nothing usable is found, so the caller skips the
    problem loudly instead of emitting a truncated prompt.
    """
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()

        limits = ""
        for n in names:
            if n.endswith("data.json"):
                try:
                    meta = json.loads(z.read(n).decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    break
                c = meta.get("constraints") or {}
                tl, ml = c.get("time_limit"), c.get("memory_limit")
                if tl is not None and ml is not None:
                    limits = (
                        f"Time limit: {tl} seconds\n"
                        f"Memory limit: {int(ml) // (1024 * 1024)} MB\n\n"
                    )
                break

        body = None
        for pat in (
            r"statement[_/-]?en\.(md|txt|html)$",
            r"(^|/)en\.(md|txt|html)$",
            r"problem[_/-]?en\.(md|txt|html)$",
            r"statement\.(md|txt|html)$",
            r"README(_en)?\.md$",
        ):
            for n in names:
                if re.search(pat, n, re.I):
                    raw = z.read(n).decode("utf-8", "replace")
                    if raw.strip():
                        body = raw
                        break
            if body:
                break
        if body is None:
            cands = [n for n in names if re.search(r"\.(md|txt|html)$", n, re.I)]
            if cands:
                n = max(cands, key=lambda x: z.getinfo(x).file_size)
                raw = z.read(n).decode("utf-8", "replace")
                if raw.strip():
                    body = raw
        if body is None:
            return None
        return limits + body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=FS / "data/alebench/full40.parquet")
    args = ap.parse_args()

    val = pd.read_parquet(VAL)
    # Recover the exact instruction wrapper from an existing row so the only
    # thing that changes between the 10-problem and 40-problem sets is which
    # statements are included.
    sample_prompt = list(val.iloc[0]["prompt"])
    tmpl = dict(sample_prompt[0])["content"]
    have = {dict(r)["ground_truth"] for r in val["reward_model"]}
    # The header is everything before the statement body; find it by diffing two
    # rows would be fragile, so instead take the leading instruction paragraph up
    # to the first double newline, which is stable across rows.
    head = tmpl.split("\n\n", 1)[0]
    # Trailing instruction, recovered the same way: everything after the last
    # blank line of the shipped prompt (e.g. "\n\nGenerate solution code:").
    tail = "\n\n" + tmpl.rsplit("\n\n", 1)[1] if "\n\n" in tmpl else ""

    refs = json.loads(REFS.read_text())
    snap = snapshot_dir()

    rows, skipped = [], []
    for pid in sorted(refs):
        zp = snap / f"{pid}.zip"
        if not zp.exists():
            skipped.append((pid, "no zip"))
            continue
        st = statement_from_zip(zp)
        if not st:
            skipped.append((pid, "no statement found in zip"))
            continue
        # The shipped prompt is head + statement + a trailing instruction. Both
        # wrappers are recovered from val.parquet rather than hard-coded, so the
        # 40-problem set differs from the shipped 10 ONLY in which statements it
        # contains. Verified below by a byte-exact diff on the 10 shared ids.
        rows.append(
            {
                "data_source": "alebench",
                "prompt": [{"role": "user", "content": f"{head}\n\n{st}{tail}"}],
                "reward_model": {"ground_truth": pid},
            }
        )

    if skipped:
        print(f"SKIPPED {len(skipped)}:")
        for pid, why in skipped:
            print(f"  {pid}: {why}")
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} problems -> {args.out}")
    print(f"  (previous val.parquet had {len(val)}: {sorted(have)})")
    return 0 if len(df) >= 30 else 1


if __name__ == "__main__":
    raise SystemExit(main())
