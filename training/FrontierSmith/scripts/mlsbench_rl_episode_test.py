#!/usr/bin/env python3
"""NO-TRAIN harness test for MLSBenchAgentLoop.

Drives 1-2 full MLS-Bench episodes against a locally running vLLM server
(token-in via /v1/completions with prompt token-id arrays), using the REAL
MLSBenchAgentLoop code path: workspace setup, XML tool-call parsing, Apptainer
test execution, mask building, and deterministic mlsbench scoring.

Verifies and prints:
  * per-turn token counts (assistant mask=1 vs tool/nudge mask=0)
  * response_mask correctness invariants
  * the computed task score (reward)

Usage (on a GPU node with vLLM already serving MODEL at --base-url):
  python scripts/mlsbench_rl_episode_test.py \
      --base-url http://127.0.0.1:PORT/v1 --model TAG \
      --tasks ml-clustering-algorithm --episodes 1

Per-episode budget tiers (same path training uses — extra_info.budget resolved
by MLSBenchAgentLoop.resolve_budget, overriding MLS_RL_MAX_STEPS/MAX_TESTS):
  --budget-tier tight|loose            inject BUDGET_TIERS[tier] from
                                       prepare_mlsbench_rl_parquet.py
  --parquet data/mlsbench_rl/train_tv1.parquet
                                       use the ACTUAL parquet row's extra_info
                                       for (task, --budget-tier) — end-to-end
                                       honest replica of a training sample
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "verl"))


def _plain(obj):
    """Recursively convert parquet/numpy round-trip values to plain python."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_plain(x) for x in obj]
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _load_budget_tiers() -> dict:
    """Import BUDGET_TIERS from prepare_mlsbench_rl_parquet.py (single source)."""
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "prepare_mlsbench_rl_parquet.py"
    spec = importlib.util.spec_from_file_location("prepare_mlsbench_rl_parquet", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BUDGET_TIERS


def build_extra_info(task: str, index: int, args) -> dict:
    """extra_info exactly as the training dataset would deliver it to the loop."""
    if args.parquet:
        import pandas as pd

        df = pd.read_parquet(args.parquet)
        for _, row in df.iterrows():
            ei = _plain(row["extra_info"])
            if ei.get("task") != task:
                continue
            tier = (ei.get("budget") or {}).get("tier")
            if args.budget_tier and tier != args.budget_tier:
                continue
            print(f"[budget ] using parquet row extra_info: {json.dumps(ei.get('budget'))} uid={ei.get('uid')}")
            return ei
        raise SystemExit(
            f"no parquet row for task={task} tier={args.budget_tier} in {args.parquet}"
        )
    if args.budget_tier:
        budget = _load_budget_tiers()[args.budget_tier]
        print(f"[budget ] injecting tier '{args.budget_tier}': {json.dumps(budget)}")
        return {"task": task, "index": index, "uid": f"{task}::{budget['tier']}", "budget": dict(budget)}
    return {"task": task, "index": index}


class FakeServerManager:
    """Minimal stand-in for AsyncLLMServerManager: token-ids in -> token-ids out
    via a vLLM OpenAI /v1/completions endpoint (skip_special_tokens=False so the
    re-encoded text roundtrips <|im_end|> etc.)."""

    def __init__(self, base_url: str, model: str, tokenizer):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer = tokenizer

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        import requests

        payload = {
            "model": self.model,
            "prompt": list(prompt_ids),
            "max_tokens": int(sampling_params.get("max_tokens", 2048)),
            "temperature": float(sampling_params.get("temperature", 1.0)),
            "skip_special_tokens": False,
        }
        top_p = sampling_params.get("top_p")
        if top_p is not None:
            payload["top_p"] = float(top_p)

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: requests.post(f"{self.base_url}/completions", json=payload, timeout=1800)
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        text = choice["text"]
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        # vLLM strips the eos token from returned text even with skip_special_tokens=False
        # in some configs; append <|im_end|> when finish_reason == "stop" and it's absent,
        # so the loop's clean-stop detection matches the real token-level server.
        if choice.get("finish_reason") == "stop":
            im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if not token_ids or token_ids[-1] != im_end:
                token_ids = token_ids + [im_end]
        return SimpleNamespace(
            token_ids=token_ids,
            log_probs=None,
            num_preempted=None,
            routed_experts=None,
            stop_reason="completed" if choice.get("finish_reason") in ("stop", "length") else "aborted",
        )


async def run_episodes(args) -> int:
    from omegaconf import OmegaConf
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.agent_loop.mlsbench_agent_loop import MLSBenchAgentLoop

    tokenizer = hf_tokenizer(args.model_path, trust_remote_code=True)
    processor = hf_processor(args.model_path, trust_remote_code=True)

    trainer_config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": args.prompt_length, "response_length": args.response_length}
            }
        }
    )
    data_config = OmegaConf.create({"apply_chat_template_kwargs": {}})

    server_manager = FakeServerManager(args.base_url, args.model, tokenizer)

    loop = MLSBenchAgentLoop(
        trainer_config=DictConfigWrap(config=trainer_config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=RLHFDataset,
        dataset_config=DictConfigWrap(config=data_config),
    )

    sampling_params = {"temperature": 1.0, "top_p": 1.0}
    ok = 0
    for ep, task in enumerate([t for t in args.tasks for _ in range(args.episodes)]):
        print(f"\n===== EPISODE {ep} task={task} tier={args.budget_tier or '(env-default)'} =====", flush=True)
        extra_info = build_extra_info(task, ep, args)
        t0 = time.time()
        out = await loop.run(sampling_params, extra_info=extra_info, raw_prompt=[
            {"role": "user", "content": f"(fallback) MLS-Bench task {task}"}
        ])
        dt = time.time() - t0

        n_p = len(out.prompt_ids)
        n_r = len(out.response_ids)
        n_mask1 = sum(out.response_mask)
        n_mask0 = n_r - n_mask1
        print(f"[episode] elapsed={dt:.1f}s reward={out.reward_score} num_turns={out.num_turns}")
        print(f"[tokens ] prompt={n_p} response={n_r} assistant(mask=1)={n_mask1} tool/nudge(mask=0)={n_mask0}")
        print(f"[extra  ] {json.dumps({k: v for k, v in out.extra_fields.items() if k.startswith('mls_')}, default=str)}")
        ef = out.extra_fields
        print(f"[budget ] enforced: steps={ef.get('mls_steps')}/{ef.get('mls_max_steps')} "
              f"tests={ef.get('mls_tests')}/{ef.get('mls_max_tests')} "
              f"tier={ef.get('mls_budget_tier')} source={ef.get('mls_budget_source')} "
              f"finished={ef.get('mls_finished_reason')}")

        # ---- invariants ----
        assert ef.get("mls_steps", 0) <= ef.get("mls_max_steps", 1 << 30), "action budget exceeded"
        assert ef.get("mls_tests", 0) <= ef.get("mls_max_tests", 1 << 30), "test budget exceeded"
        if args.budget_tier:
            assert ef.get("mls_budget_source") == "extra_info", "budget not resolved from extra_info"
        assert len(out.response_mask) == len(out.response_ids), "mask/ids length mismatch"
        assert n_r <= args.response_length, "response over budget"
        assert n_p <= args.prompt_length, "prompt over budget"
        # mask=1 segments must decode to the assistant text (spot-check: first segment
        # starts at position 0 — the first generated turn directly follows the prompt)
        if n_r > 1:
            assert out.response_mask[0] == 1, "first response token must be assistant-generated"
        # tool segments must be flanked by assistant segments (never trailing mask=0
        # unless episode ended on a delta append cut)
        print("[mask   ] segments:", _segments(out.response_mask))
        text_tail = tokenizer.decode(out.response_ids[-min(300, n_r):])
        print(f"[tail   ] ...{text_tail[-1200:]}")
        if out.reward_score is not None:
            ok += 1
    print(f"\nDONE: {ok} episodes completed with a reward value.")
    return 0


def _segments(mask: list[int]) -> list[str]:
    segs, cur, cnt = [], None, 0
    for m in mask:
        if m == cur:
            cnt += 1
        else:
            if cur is not None:
                segs.append(f"{'A' if cur == 1 else 'T'}{cnt}")
            cur, cnt = m, 1
    if cur is not None:
        segs.append(f"{'A' if cur == 1 else 'T'}{cnt}")
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "Qwen3.5-9B"))
    ap.add_argument("--tasks", nargs="+", default=["ml-clustering-algorithm"])
    ap.add_argument("--episodes", type=int, default=1, help="episodes per task")
    ap.add_argument("--budget-tier", choices=["tight", "loose"], default=None,
                    help="per-episode budget tier via extra_info.budget (overrides env knobs)")
    ap.add_argument("--parquet", default=None,
                    help="use the real parquet row's extra_info for (task, --budget-tier)")
    ap.add_argument("--prompt-length", type=int, default=int(os.environ.get("MAX_PROMPT_LENGTH", "14336")))
    ap.add_argument("--response-length", type=int, default=int(os.environ.get("MAX_RESPONSE_LENGTH", "14336")))
    args = ap.parse_args()
    return asyncio.run(run_episodes(args))


if __name__ == "__main__":
    raise SystemExit(main())
