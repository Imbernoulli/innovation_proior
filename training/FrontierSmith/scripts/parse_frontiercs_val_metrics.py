#!/usr/bin/env python3
"""Parse Frontier-CS validation metrics from VERL trainer log output.

Usage:
  python scripts/parse_frontiercs_val_metrics.py <log_file>
  cat log.txt | python scripts/parse_frontiercs_val_metrics.py -

Reads stdin or file, extracts val-core/frontiercs/reward/* metrics, prints as CSV row.
"""

import re
import sys


def parse_metrics(text: str) -> dict[str, str]:
    """Extract Score@1 / Avg@5 / Score@5 from log text.

    PREFERS the leaderboard-faithful official metrics when present
    ("METRIC official/frontiercs/score_at_1 / avg_at_5 / score_at_5", emitted by
    eval_qwen35_base_vllm_request.py's _official_research_metrics, which mirrors
    Frontier-CS batch/state.py). Those are the numbers the leaderboard reports:
      - Score@1 = variant-0 score averaged over problems (NOT mean@5),
      - Score@5 = TRUE per-problem max over 5 (NOT the bootstrap best@5/mean).

    Falls back to the older process_validation_metrics keys (mean@1/mean@5/
    best@5/mean) for backward compatibility with logs produced before the
    official-metrics block was added -- but note best@5/mean is a bootstrap
    estimate biased low relative to the official Score@5.
    """
    result = {}
    # (key, [preferred official pattern, legacy fallback pattern(s)...])
    patterns = [
        (
            "score_at_1",
            [
                r"['\"]?official/frontiercs/score_at_1['\"]?\s*:\s*([0-9.]+)",
                # legacy: mean@1 (only emitted when n_samples==1)
                r"['\"]?val-core/frontiercs/reward/mean@1['\"]?\s*:\s*([0-9.]+)",
            ],
        ),
        (
            "avg_score_at_5",
            [
                r"['\"]?official/frontiercs/avg_at_5['\"]?\s*:\s*([0-9.]+)",
                r"['\"]?val-core/frontiercs/reward/mean@5['\"]?\s*:\s*([0-9.]+)",
            ],
        ),
        (
            "best_score_at_5",
            [
                r"['\"]?official/frontiercs/score_at_5['\"]?\s*:\s*([0-9.]+)",
                # legacy: prefer the TRUE max (oracle_best@5) over the bootstrap best@5
                r"['\"]?val-core/frontiercs/reward/oracle_best@5['\"]?\s*:\s*([0-9.]+)",
                r"['\"]?val-core/frontiercs/reward/best@5/mean['\"]?\s*:\s*([0-9.]+)",
            ],
        ),
    ]
    for key, pats in patterns:
        result[key] = ""
        for pat in pats:
            m = re.search(pat, text)
            if m:
                result[key] = m.group(1)
                break
    return result


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: parse_frontiercs_val_metrics.py <log_file>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    if path == "-":
        text = sys.stdin.read()
    else:
        with open(path) as f:
            text = f.read()
    metrics = parse_metrics(text)
    print(f"{metrics.get('score_at_1', '')},{metrics.get('avg_score_at_5', '')},{metrics.get('best_score_at_5', '')}")


if __name__ == "__main__":
    main()
