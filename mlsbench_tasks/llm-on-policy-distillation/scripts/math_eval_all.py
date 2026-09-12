"""Unified eval driver: load the student checkpoint with vLLM once, then
run GSM8K, MATH-500, and AMC23 sequentially. Saves the model-load overhead
that the previous one-eval-per-SLURM-job layout paid three times.

Emits one EVAL_METRICS line per benchmark, parsable by the same parser.

Usage (inside container):
    python math_eval_all.py
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd


_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_LAST_NUMBER_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*$")


def _extract_answer(text: str) -> str:
    if not text:
        return ""
    m = list(_BOXED_RE.finditer(text))
    if m:
        return m[-1].group(1).strip()
    last_line = text.strip().splitlines()[-1].strip() if text.strip() else ""
    m = _LAST_NUMBER_RE.search(last_line)
    if m:
        return m.group(1).replace(",", "")
    return last_line


def _normalize(s: str) -> str:
    return re.sub(r"[\s\\\$,]", "", str(s))


def _canonical_number(s: str):
    s = _normalize(s)
    if not s:
        return None
    try:
        if "/" in s and s.count("/") == 1:
            a, b = s.split("/")
            v = float(a) / float(b)
        else:
            v = float(s)
        if v.is_integer():
            return str(int(v))
        return f"{v:.10g}"
    except (ValueError, ZeroDivisionError):
        return None


def _grade(pred: str, gold: str) -> bool:
    if not pred:
        return False
    p_num, g_num = _canonical_number(pred), _canonical_number(gold)
    if p_num is not None and g_num is not None and p_num == g_num:
        return True
    if _normalize(pred) == _normalize(gold) and _normalize(pred) != "":
        return True
    try:
        from math_verify import parse as mv_parse, verify as mv_verify  # type: ignore
        gold_expr = mv_parse(f"\\boxed{{{gold}}}")
        pred_expr = mv_parse(f"\\boxed{{{pred}}}")
        return bool(mv_verify(gold_expr, pred_expr))
    except Exception:
        return False


# (label, parquet path, max_new_tokens, n_samples, temperature)
#   - gsm8k / math500 are large enough that greedy@1 is stable.
#   - amc23 only has 40 problems, so we report avg@8 with temperature=0.6 —
#     standard math-reasoning eval protocol (e.g. Qwen2.5-Math, DeepSeek-Math
#     report avg@8/16/32 for AMC/AIME).
EVAL_SPECS = [
    ("gsm8k",   "/data/datasets/gsm8k_test.parquet",   1024, 1, 0.0),
    ("math500", "/data/datasets/math500_test.parquet", 1536, 1, 0.0),
    # AMC23: avg@8 with longer cap. Empirically the post-distillation 0.5B
    # student hits the 2048 cap on essentially every AMC sample (mean ~2017
    # tokens per sample with 2048 cap), so reasoning is being truncated
    # before the boxed answer. Bumped to 4096 to give room.
    ("amc",     "/data/datasets/amc23_test.parquet",   4096, 8, 0.6),
]


def main():
    ckpt = os.environ.get("CKPT") or (os.environ.get("OUTPUT_DIR", "/tmp") + "/student_ckpt")
    seed = int(os.environ.get("SEED", 42))
    print(f"checkpoint: {ckpt}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'user' %}Question: {{ m['content'] }}\nAnswer: "
            "{% elif m['role'] == 'assistant' %}{{ m['content'] }}{{ eos_token }}"
            "{% endif %}{% endfor %}"
        )

    # Pick a max_model_len that fits the longest planned eval (AMC needs 4096
    # generation + a few-hundred-token prompt).
    max_model_len = max(spec[2] for spec in EVAL_SPECS) + 1024
    max_model_len = max(max_model_len, 8192)

    llm = LLM(
        model=ckpt,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
    )

    for label, dataset_path, max_new_tokens, n_samples, temperature in EVAL_SPECS:
        print(
            f"\n=== {label}: {dataset_path} (max_new_tokens={max_new_tokens}, n_samples={n_samples}, T={temperature}) ===",
            flush=True,
        )
        df = pd.read_parquet(dataset_path)
        print(f"loaded {len(df)} problems for {label}", flush=True)

        prompts = []
        for problem in df["problem"].tolist():
            msgs = [{
                "role": "user",
                "content": problem + "\n\nPlease place the final answer inside \\boxed{}.",
            }]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        sp = SamplingParams(
            temperature=temperature,
            top_p=1.0 if temperature == 0.0 else 0.95,
            max_tokens=max_new_tokens,
            n=n_samples,
            seed=seed,
        )
        outputs = llm.generate(prompts, sp)

        # avg@n: average accuracy across n_samples per problem, then averaged over problems.
        per_problem_correct = []
        for problem, gold, out in zip(df["problem"].tolist(), df["answer"].tolist(), outputs):
            sample_completions = [s.text for s in out.outputs]
            sample_hits = sum(1 for c in sample_completions if _grade(_extract_answer(c), gold))
            per_problem_correct.append(sample_hits / max(len(sample_completions), 1))

        n_total = len(df)
        acc = sum(per_problem_correct) / max(n_total, 1)
        # n_correct is an int approximation for compatibility (rounded effective hits).
        n_correct = int(round(sum(per_problem_correct)))
        suffix = f"@{n_samples}" if n_samples > 1 else ""
        print(
            f"EVAL_METRICS {label}_accuracy{suffix}={acc:.6f} "
            f"{label}_n_correct{suffix}={n_correct} {label}_n_total{suffix}={n_total} "
            f"{label}_n_samples={n_samples}",
            flush=True,
        )
        # Also emit the plain ``<label>_accuracy`` alias so the parser/leaderboard
        # column name stays unchanged whether we run greedy@1 or avg@n.
        if n_samples > 1:
            print(
                f"EVAL_METRICS {label}_accuracy={acc:.6f} {label}_n_correct={n_correct} {label}_n_total={n_total}",
                flush=True,
            )


if __name__ == "__main__":
    main()
