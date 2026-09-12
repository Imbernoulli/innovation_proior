"""On-policy distillation training driver for llm-on-policy-distillation.

Loads the OpenR1-Math prompt parquet, wraps each prompt in ChatML (matching
TRL's DataCollatorForChatML expectations), and runs GKDTrainer.

The student model is saved to <output_dir> at the end of training; the
eval scripts (eval_gsm8k.sh / eval_math500.sh / eval_amc.sh) consume it.

Hyperparameters mirror TRL's reference GKD example, with on-policy mixing
turned on (lmbda=0.5 by default; baselines override via flags).
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.gkd import GKDConfig, GKDTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--teacher_model_name_or_path", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--logging_steps", type=int, default=20)
    return p.parse_args()


def _to_chatml(row):
    prompt = row["prompt"] if isinstance(row.get("prompt"), str) else str(row.get("prompt") or "")
    # Use the dataset solution as the off-policy assistant turn. DataCollatorForChatML
    # labels only the assistant tokens, so an empty content would make the
    # off-policy (1-lmbda) fraction of batches train only on EOS. With the
    # solution populated, the off-policy half is a meaningful supervised target.
    solution = row.get("solution")
    if not isinstance(solution, str) or not solution.strip():
        # Fallback: use the bare numeric answer if "solution" missing.
        solution = str(row.get("answer") or "").strip()
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": solution},
        ]
    }


def main():
    args = parse_args()
    print(f"args = {vars(args)}", flush=True)

    # ── tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Qwen2.5-0.5B base has no chat_template; install a minimal one so the
    # ChatML collator can render user/assistant turns deterministically.
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'user' %}Question: {{ m['content'] }}\nAnswer: "
            "{% elif m['role'] == 'assistant' %}{{ m['content'] }}{{ eos_token }}"
            "{% endif %}{% endfor %}"
        )

    # ── dataset ────────────────────────────────────────────────────────
    df = pd.read_parquet(args.dataset_path)
    print(f"loaded {len(df)} prompt rows from {args.dataset_path}", flush=True)
    drop_cols = list(df.columns)
    ds = Dataset.from_pandas(df, preserve_index=False).map(_to_chatml, remove_columns=drop_cols)

    # ── config ─────────────────────────────────────────────────────────
    cfg = GKDConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        beta=args.beta,
        lmbda=args.lmbda,
        temperature=args.temperature,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        teacher_model_init_kwargs={"dtype": "bfloat16"},
        use_liger_kernel=False,
        disable_dropout=True,
    )

    trainer = GKDTrainer(
        model=args.model_name_or_path,
        teacher_model=args.teacher_model_name_or_path,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.train()

    if trainer.accelerator.is_main_process:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))
        print(f"TRAIN_DONE checkpoint saved to {out}", flush=True)


if __name__ == "__main__":
    main()
