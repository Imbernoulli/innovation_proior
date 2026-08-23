#!/usr/bin/env python3
"""Standalone FrontierCS scoring diagnostic / re-scorer.

Loads samples from an existing samples.jsonl (records must carry a ``text``
field and a ``ground_truth`` problem id), extracts the C++ via the OFFICIAL
``extract_cpp_code``, and runs the OFFICIAL ``AlgorithmicLocalRunner`` against a
running judge -- capturing the FULL judge response (status, score, per-case
verdicts, compile/runtime stderr) so we can see exactly why a score is 0.

Assumes the judge (go-judge + node server) is already running at --judge-url.
Start it with scripts/start_frontiercs_judge_hybrid.sh.

Usage:
  python scripts/diag_frontiercs_rescore.py \
      --samples outputs/frontiercs_eval_qwen35_9b_official_vllm/samples.jsonl \
      --judge-url http://127.0.0.1:8082 \
      --limit 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_FRONTIERCS_ROOT = PROJECT_ROOT / ".cache" / "Frontier-CS-official"
LOCAL_FRONTIERCS_ROOT = PROJECT_ROOT / "Frontier-CS"
if OFFICIAL_FRONTIERCS_ROOT.exists():
    sys.path.insert(0, str(OFFICIAL_FRONTIERCS_ROOT))
    sys.path.insert(0, str(OFFICIAL_FRONTIERCS_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from algorithmic.scripts.generate_solutions import extract_cpp_code as official_extract_cpp_code  # noqa: E402
from frontier_cs.runner.algorithmic_local import AlgorithmicLocalRunner  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Robust JSONL loader: tolerant of huge records / accidental newlines."""
    recs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        for line in f:
            buf += line
            try:
                recs.append(json.loads(buf))
                buf = ""
            except json.JSONDecodeError:
                continue
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:8082")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument(
        "--only-gt",
        default=None,
        help="Comma-separated ground_truth ids to restrict to (e.g. 109,112,123).",
    )
    ap.add_argument("--out", type=Path, default=None, help="Optional JSONL to write rescored rows.")
    args = ap.parse_args()

    runner = AlgorithmicLocalRunner(
        judge_url=args.judge_url,
        base_dir=LOCAL_FRONTIERCS_ROOT,
        auto_start=False,
    )
    print(f"judge_url={runner.judge_url} base_dir={runner.base_dir}", flush=True)
    print(f"judge available: {runner._is_judge_available()}", flush=True)
    probs = runner.list_problems()
    print(f"judge reports {len(probs)} problems available", flush=True)

    recs = _load_jsonl(args.samples)
    only = set(args.only_gt.split(",")) if args.only_gt else None
    if only:
        recs = [r for r in recs if str(r.get("ground_truth")) in only]

    # Prefer one sample per problem id, deterministic order.
    seen: set[str] = set()
    picked: list[dict[str, Any]] = []
    for r in recs:
        gt = str(r.get("ground_truth"))
        if gt in seen:
            continue
        if "text" not in r or "```cpp" not in r.get("text", ""):
            continue
        seen.add(gt)
        picked.append(r)
        if len(picked) >= args.limit:
            break

    out_f = args.out.open("w", encoding="utf-8") if args.out else None
    print(f"\nRe-scoring {len(picked)} samples\n", flush=True)
    print(f"{'gt':>6} {'old':>8} {'new':>10} {'status':>10}  detail", flush=True)
    n_nonzero = 0
    for r in picked:
        gt = str(r["ground_truth"])
        old_score = (r.get("metrics") or {}).get("score")
        code = official_extract_cpp_code(r["text"])
        if not code:
            print(f"{gt:>6} {str(old_score):>8} {'EXTRACT0':>10} {'-':>10}  extract_cpp_code returned empty", flush=True)
            continue
        res = runner.evaluate(gt, code)
        new_score = float(res.score or 0.0) if res.success else 0.0
        if new_score:
            n_nonzero += 1
        meta = res.metadata or {}
        # Pull out a compact detail string from the raw judge payload.
        detail_bits = []
        if not res.success:
            detail_bits.append(f"FAIL:{res.message}")
        cases = meta.get("cases") or meta.get("results") or []
        if isinstance(cases, list) and cases:
            verdicts = [c.get("status") or c.get("verdict") for c in cases if isinstance(c, dict)]
            detail_bits.append(f"cases={len(cases)} verdicts={verdicts[:6]}")
        if meta.get("message"):
            detail_bits.append(f"msg={str(meta['message'])[:120]}")
        for k in ("compileError", "compile_error", "stderr", "error", "logs"):
            if meta.get(k):
                detail_bits.append(f"{k}={str(meta[k])[:160]}")
                break
        detail = " | ".join(detail_bits)
        print(
            f"{gt:>6} {str(old_score):>8} {new_score:>10.4f} {res.status.value if hasattr(res.status,'value') else res.status:>10}  {detail}",
            flush=True,
        )
        if out_f:
            out_f.write(
                json.dumps(
                    {
                        "ground_truth": gt,
                        "old_score": old_score,
                        "new_score": new_score,
                        "success": res.success,
                        "status": res.status.value if hasattr(res.status, "value") else str(res.status),
                        "score_unbounded": res.score_unbounded,
                        "judge_status": meta.get("status"),
                    }
                )
                + "\n"
            )
    if out_f:
        out_f.close()
    print(f"\nRESULT: {n_nonzero}/{len(picked)} samples now score NONZERO", flush=True)


if __name__ == "__main__":
    main()
