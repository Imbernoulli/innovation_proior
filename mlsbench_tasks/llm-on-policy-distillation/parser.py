"""Output parser for the llm-on-policy-distillation task.

There are two test_cmds:

    train  -> GKDTrainer training script (no leaderboard metric)
    eval   -> math_eval_all.py: GSM8K (greedy@1, 1319 problems) + MATH-500
              (greedy@1, 500 problems) + AMC23 (avg@8 with temperature=0.6,
              40 problems) — all three benchmarks served from a single
              vLLM-loaded student checkpoint.

Each training step prints:
    TRAIN_METRICS step=N loss=X lr=Y lmbda=Z beta=W

Each evaluation script prints exactly one line at the end:
    EVAL_METRICS <label>_accuracy=<float> <label>_n_correct=<int> <label>_n_total=<int>

The parser surfaces the last training updates as agent feedback and
emits per-benchmark accuracy as leaderboard metrics.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\(B")
_KV_RE = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def _clean(line: str) -> str:
    return _ANSI_RE.sub("", line).strip()


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        lines = [_clean(l) for l in raw_output.splitlines()]
        feedback_parts: list[str] = []
        metrics: dict = {}

        # TRAIN_METRICS lines (training script only)
        train_lines = [l for l in lines if l.startswith("TRAIN_METRICS ")]
        if train_lines:
            tail = train_lines[-5:]
            feedback_parts.append(
                f"Training progress ({cmd_label}, last {len(tail)} of {len(train_lines)} updates):\n"
                + "\n".join(tail)
            )
            last_kv = dict(_KV_RE.findall(train_lines[-1]))
            if "loss" in last_kv:
                try:
                    metrics["final_train_loss"] = float(last_kv["loss"])
                except ValueError:
                    pass

        # EVAL_METRICS lines (eval scripts only; usually just one at the end)
        eval_lines = [l for l in lines if l.startswith("EVAL_METRICS ")]
        for line in eval_lines:
            for k, v in _KV_RE.findall(line):
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass
        if eval_lines:
            feedback_parts.append(
                f"Evaluation result ({cmd_label}):\n" + "\n".join(eval_lines[-2:])
            )

        # ERROR lines surface as warnings
        error_lines = [
            l for l in lines
            if l.startswith("ERROR ") or l.startswith("Traceback") or "CUDA out of memory" in l
        ]
        if error_lines:
            feedback_parts.append(
                "Errors detected:\n" + "\n".join(error_lines[:10])
            )

        # Fallback: dump tail of raw output if nothing parsed
        if not feedback_parts:
            tail = "\n".join(lines[-30:])
            feedback_parts.append(
                f"No TRAIN_METRICS or EVAL_METRICS lines found in '{cmd_label}' output. "
                f"Last 30 lines:\n{tail}"
            )

        return ParseResult(
            feedback="\n\n".join(feedback_parts),
            metrics=metrics,
        )
