# Copyright 2026 FrontierSmith MLS-Bench RL integration.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MLSBenchAgentLoop: one MLS-Bench agentic episode per rollout, token-in-token-out.

ADDITIVE module: only imported when a run sets
  actor_rollout_ref.rollout.agent.agent_loop_config_path=<yaml referencing this class>
and the parquet rows carry agent_name="mlsbench_agent". Existing single-turn runs
never import this file and are byte-identical.

Design (Option A — native AgentLoop + per-episode environment subprocess):
  * Generation is token-in-token-out through verl's AsyncLLMServerManager, so
    response_mask covers EXACTLY the sampled assistant tokens; tool responses /
    nudges are appended as mask-0 token deltas rendered with the Qwen3.5 chat
    template's exact literals (apply_chat_template rejects tool-only lists).
  * Tool calls are parsed from the decoded assistant text with a Qwen3.5
    XML-format parser (<tool_call><function=NAME><parameter=K>V</parameter>...),
    schema-aware type coercion — mirrors vLLM's qwen3_coder parser.
  * The MLS-Bench environment (workspace, edit/test/submit/undo, Apptainer
    execution, leaderboard, deterministic scoring) runs UNCHANGED inside a
    per-episode subprocess (scripts/mlsbench_rl_episode_worker.py) under the
    proven eval conda python. Process-per-episode is REQUIRED: task
    mid_edit.py/parser.py files `import dgp` from per-task holdout dirs and
    poison sys.modules across tasks (verified), exactly why the eval harness
    also runs one process per task. It also gives a clean kill for hung tests.
  * Reward: worker force-submits the last test if the agent didn't (eval's
    auto-submit semantics), then computes the deterministic [0,1] task score
    via mlsbench.scoring.evaluate.evaluate_task keyed by a per-episode unique
    model name. Any failure => reward 0.0 (fail-soft). Scores land in
    AgentLoopOutput.reward_score -> rm_scores, so no reward-fn is needed.

Env knobs (all optional):
  MLS_RL_MLSBENCH_ROOT   MLS-Bench checkout (DEV root; default MLS-Bench-dev)
  MLS_RL_WORKER_PYTHON   python for the episode worker (default eval conda python)
  MLS_RL_WORKER_SCRIPT   path to mlsbench_rl_episode_worker.py
  MLS_RL_DATA_ROOT       data_root for task containers
  MLS_RL_OUTPUT_BASE     base dir for saves/workspaces/episode_logs
  MLS_RL_MAX_STEPS       DEFAULT agent action budget per episode (default 8)
  MLS_RL_MAX_TESTS       DEFAULT test() budget per episode (default 1)
                         Both are FALLBACKS: a per-row extra_info["budget"]
                         ({"max_tests":N,"max_actions":N,"tier":S}, written by
                         scripts/prepare_mlsbench_rl_parquet.py) OVERRIDES them
                         per episode (see resolve_budget). Rows without a
                         budget keep the env-knob behavior unchanged.
  MLS_RL_USE_REPLACE     1 => str_replace edit schema (devfix eval口径; default 1)
  MLS_RL_MAX_CONC_TESTS  concurrent Apptainer test() executions per worker process (default 4)
  MLS_RL_EPISODE_TIMEOUT hard wall-clock seconds per episode (default 2700)
  MLS_RL_FINALIZE_TIMEOUT seconds for force-submit + scoring (default 300)
  MLS_RL_NUDGE_LIMIT     max "please call a tool" nudges (default 2)
  MLS_RL_MAX_TOOL_RESPONSE_CHARS  tool result truncation (default 8000)
  MLS_RL_KEEP_WORKSPACE  1 => keep episode workspaces (default 0)
"""

import asyncio
import importlib.util
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_DEFAULT_FS_ROOT = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _load_worker_client_cls():
    """Import WorkerClient from the worker script file (stdlib-only import)."""
    script = os.environ.get(
        "MLS_RL_WORKER_SCRIPT", os.path.join(_DEFAULT_FS_ROOT, "scripts", "mlsbench_rl_episode_worker.py")
    )
    spec = importlib.util.spec_from_file_location("mlsbench_rl_episode_worker", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WorkerClient, script


# ---------------------------------------------------------------------------
# Qwen3.5 XML tool-call parser (chat_template.jinja "<function=...>" format)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def _strip_one_newline(v: str) -> str:
    """Template renders '<parameter=K>\\n' + value + '\\n</parameter>' — strip exactly one each side."""
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def _coerce(value: str, ptype: Optional[str]):
    if ptype == "integer":
        try:
            return int(float(value.strip()))
        except ValueError:
            return value
    if ptype == "number":
        try:
            return float(value.strip())
        except ValueError:
            return value
    if ptype == "boolean":
        return value.strip().lower() in ("1", "true", "yes", "on")
    if ptype in ("object", "array"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


_THINK_SPAN_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(turn_text: str) -> str:
    """Return only the post-reasoning content of one assistant turn.

    The generation prompt ends with "<think>\\n" (the opening tag lives in the
    PROMPT delta), so the sampled turn text is reasoning up to the first
    "</think>"; content follows it. Mirrors vLLM's qwen3 reasoning parser
    (split on FIRST </think>) — the step inference engines apply between decode
    and tool parsing, which verl's agent loop is missing (verl issues #4757 /
    #6424: tool-call syntax INSIDE <think> is otherwise executed as a real
    call). If the turn never closes its reasoning, the whole turn is reasoning
    → no content. Any residual complete <think>…</think> spans in the content
    are stripped too (belt and braces, per the #6434 fix direction).
    """
    if "</think>" in turn_text:
        content = turn_text.split("</think>", 1)[1]
    else:
        return ""
    return _THINK_SPAN_RE.sub("", content)


def parse_qwen_xml_tool_calls(text: str, schemas_by_name: dict[str, dict]) -> list[tuple[str, dict]]:
    """Parse Qwen3.5 XML-style tool calls out of decoded assistant text.

    Returns [(tool_name, args_dict), ...]; schema-aware type coercion via the
    Anthropic-style input_schema of each tool. NOTE: callers should pass
    post-reasoning content (see strip_reasoning) so hypothetical tool calls
    written inside <think> are never executed.
    """
    calls: list[tuple[str, dict]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        body = m.group(2)
        schema = schemas_by_name.get(name)
        props = ((schema or {}).get("input_schema") or {}).get("properties", {})
        args: dict = {}
        for pm in _PARAM_RE.finditer(body):
            key = pm.group(1).strip()
            raw = _strip_one_newline(pm.group(2))
            args[key] = _coerce(raw, (props.get(key) or {}).get("type"))
        calls.append((name, args))
    return calls


def anthropic_to_openai_tools(schemas: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s.get("input_schema", {}),
            },
        }
        for s in schemas
    ]


NUDGE_TEXT = "Please use one of the available tools (edit, test, or undo) to continue working on the task."


def resolve_budget(extra_info: Optional[dict], default_max_steps: int, default_max_tests: int) -> dict:
    """Resolve the per-episode action/test budget.

    Per-row ``extra_info["budget"]`` — written by
    scripts/prepare_mlsbench_rl_parquet.py as
    ``{"max_tests": int, "max_actions": int, "tier": "tight"|"loose"}`` —
    OVERRIDES the global env knobs MLS_RL_MAX_STEPS / MLS_RL_MAX_TESTS.
    The env knobs remain the defaults for rows without a budget (backward
    compatible), and per-key fallbacks apply for partial budget dicts.

    Tolerates parquet round-trip artifacts: JSON-string budgets, numpy scalar
    values, and mapping-like structs.

    Returns {"max_steps": int, "max_tests": int, "tier": str|None,
             "source": "extra_info"|"env"}.
    """
    budget = (extra_info or {}).get("budget")
    if isinstance(budget, (bytes, str)):
        try:
            budget = json.loads(budget)
        except (json.JSONDecodeError, UnicodeDecodeError):
            budget = None
    if budget is None or not hasattr(budget, "get"):
        return {
            "max_steps": int(default_max_steps),
            "max_tests": int(default_max_tests),
            "tier": None,
            "source": "env",
        }

    def _pick(keys: tuple, default: int) -> int:
        for k in keys:
            v = budget.get(k)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return int(default)

    tier = budget.get("tier")
    return {
        "max_steps": _pick(("max_actions", "max_steps"), default_max_steps),
        "max_tests": _pick(("max_tests",), default_max_tests),
        "tier": str(tier) if tier is not None else None,
        "source": "extra_info",
    }


class _EpisodeState:
    def __init__(self):
        self.client = None
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.assistant_turns = 0
        self.user_turns = 0
        self.nudges = 0
        self.turn_log: list[dict] = []
        self.timed_out = False
        self.error: Optional[str] = None
        self.reward: float = 0.0
        self.score_detail: Optional[dict] = None
        self.gen_seconds = 0.0
        self.tool_seconds = 0.0
        self.model_name = ""
        self.task_name = ""
        self.finished_reason = ""
        self.steps = 0
        self.tests = 0
        self.done = False
        self.log_dir = ""
        # per-episode budget (resolved from extra_info.budget or env defaults)
        self.max_steps = 0
        self.max_tests = 0
        self.budget_tier: Optional[str] = None
        self.budget_source = "env"


@register("mlsbench_agent")
class MLSBenchAgentLoop(AgentLoopBase):
    """One MLS-Bench episode per rollout sample. See module docstring."""

    _test_semaphore: Optional[asyncio.Semaphore] = None

    def __init__(self, trainer_config, server_manager, tokenizer, processor, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

        self.max_steps = _env_int("MLS_RL_MAX_STEPS", 8)
        self.max_tests = _env_int("MLS_RL_MAX_TESTS", 1)
        self.use_replace = _env_flag("MLS_RL_USE_REPLACE", True)
        self.episode_timeout = _env_int("MLS_RL_EPISODE_TIMEOUT", 2700)
        self.finalize_timeout = _env_int("MLS_RL_FINALIZE_TIMEOUT", 300)
        self.nudge_limit = _env_int("MLS_RL_NUDGE_LIMIT", 2)
        self.max_tool_response_chars = _env_int("MLS_RL_MAX_TOOL_RESPONSE_CHARS", 8000)
        self.keep_workspace = _env_flag("MLS_RL_KEEP_WORKSPACE", False)

        base = os.environ.get("MLS_RL_OUTPUT_BASE", "/tmp/mls_rl")
        self.save_path = os.environ.get("MLS_RL_SAVE_PATH", os.path.join(base, "saves"))
        self.workspace_root = os.environ.get("MLS_RL_WORKSPACE_ROOT", os.path.join(base, "workspaces"))
        self.episode_log_dir = os.environ.get("MLS_RL_EPISODE_LOG_DIR", os.path.join(base, "episode_logs"))
        self.data_root = os.environ.get(
            "MLS_RL_DATA_ROOT", "/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data"
        )
        self.worker_python = os.environ.get("MLS_RL_WORKER_PYTHON", "/home/bl3615/miniconda3/bin/python")
        self.worker_client_cls, self.worker_script = _load_worker_client_cls()

        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_id = self.tokenizer.eos_token_id

    @classmethod
    def _get_test_semaphore(cls) -> asyncio.Semaphore:
        if cls._test_semaphore is None:
            cls._test_semaphore = asyncio.Semaphore(_env_int("MLS_RL_MAX_CONC_TESTS", 4))
        return cls._test_semaphore

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _spawn_worker(self, task_name: str, model_name: str, max_steps: int, max_tests: int):
        os.makedirs(self.episode_log_dir, exist_ok=True)
        stderr_path = os.path.join(
            self.episode_log_dir, f"worker_{task_name}_{model_name.rsplit('-', 1)[-1]}.log"
        )
        client = self.worker_client_cls(
            worker_python=self.worker_python,
            worker_script=self.worker_script,
            env={
                "MLS_RL_MLSBENCH_ROOT": os.environ.get(
                    "MLS_RL_MLSBENCH_ROOT", "/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev"
                ),
                "MLSBENCH_SCHEDULER_MANAGED": "1",
                "MLSBENCH_NO_PREBUILT": "1",
            },
            stderr_path=stderr_path,
        )
        init = client.call(
            cmd="init",
            task=task_name,
            model_name=model_name,
            max_steps=max_steps,
            max_tests=max_tests,
            save_path=self.save_path,
            workspace_root=self.workspace_root,
            data_root=self.data_root,
            use_replace=self.use_replace,
        )
        return client, init

    def _truncate_tool_response(self, text: str) -> str:
        limit = self.max_tool_response_chars
        if len(text) <= limit:
            return text
        half = limit // 2
        return text[:half] + "\n...(truncated)...\n" + text[-half:]

    def _delta_ids(self, user_text: str, wrap_tool_response: bool) -> list[int]:
        """Render a user/tool turn + generation prompt with the Qwen3.5 template's exact literals."""
        body = f"<tool_response>\n{user_text}\n</tool_response>" if wrap_tool_response else user_text
        delta_text = f"\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        return self.tokenizer.encode(delta_text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info") or {}
        task_name = extra_info.get("task") or extra_info.get("task_name")
        assert task_name, f"mlsbench_agent requires extra_info.task, got keys={list(extra_info.keys())}"

        state = _EpisodeState()
        state.task_name = str(task_name)
        state.model_name = f"vllm/rl-{uuid.uuid4().hex[:12]}"

        # Per-episode budget: extra_info.budget (parquet row) overrides env knobs.
        budget = resolve_budget(extra_info, self.max_steps, self.max_tests)
        state.max_steps = budget["max_steps"]
        state.max_tests = budget["max_tests"]
        state.budget_tier = budget["tier"]
        state.budget_source = budget["source"]

        t_start = time.time()
        try:
            await asyncio.wait_for(self._episode(state, sampling_params), timeout=self.episode_timeout)
        except asyncio.TimeoutError:
            state.timed_out = True
            state.finished_reason = state.finished_reason or "episode_timeout"
            logger.warning("[mls-rl] episode timeout (%ss) task=%s", self.episode_timeout, state.task_name)
        except Exception as e:  # noqa: BLE001
            state.error = repr(e)
            state.finished_reason = state.finished_reason or "error"
            logger.exception("[mls-rl] episode failed task=%s: %r", state.task_name, e)

        # ---- finalize + score in the worker (fail-soft, bounded) ----
        if state.client is not None:
            if state.timed_out:
                # worker may be stuck inside a test — kill; no score possible
                await self.loop.run_in_executor(None, state.client.kill)
                state.reward, state.score_detail = 0.0, {"reason": "episode_timeout"}
            else:
                try:
                    fin = await asyncio.wait_for(
                        self.loop.run_in_executor(None, lambda: state.client.call(cmd="finalize")),
                        timeout=self.finalize_timeout,
                    )
                    state.reward = float(fin.get("reward") or 0.0)
                    state.score_detail = fin.get("detail")
                except Exception as e:  # noqa: BLE001
                    logger.warning("[mls-rl] finalize failed task=%s: %r -> reward 0.0", state.task_name, e)
                    state.reward, state.score_detail = 0.0, {"reason": f"finalize_error: {e!r}"}
                try:
                    await self.loop.run_in_executor(
                        None,
                        lambda: state.client.call(cmd="cleanup", keep_workspace=self.keep_workspace),
                    )
                except Exception:  # noqa: BLE001
                    pass
                await self.loop.run_in_executor(None, state.client.quit)
        else:
            state.reward, state.score_detail = 0.0, {"reason": "worker_init_failed"}

        # ---- fallback trajectory if init failed before any tokens ----
        if not state.prompt_ids:
            try:
                messages = list(kwargs["raw_prompt"])
                state.prompt_ids = await self.apply_chat_template(messages)
            except Exception:  # noqa: BLE001
                state.prompt_ids = self.tokenizer.encode("mlsbench episode failed", add_special_tokens=False)
        if not state.response_ids:
            state.response_ids = [self.im_end_id if self.im_end_id is not None else (self.eos_id or 0)]
            state.response_mask = [0]

        if len(state.prompt_ids) > self.prompt_length:
            logger.warning(
                "[mls-rl] prompt too long (%d > %d) task=%s — truncating left",
                len(state.prompt_ids), self.prompt_length, state.task_name,
            )
            state.prompt_ids = state.prompt_ids[-self.prompt_length:]
        response_ids = state.response_ids[: self.response_length]
        response_mask = state.response_mask[: self.response_length]

        elapsed = time.time() - t_start
        self._write_episode_log(state, elapsed)

        output = AgentLoopOutput(
            prompt_ids=state.prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            multi_modal_data={},
            reward_score=state.reward,
            num_turns=state.assistant_turns + state.user_turns + 1,
            metrics={"generate_sequences": state.gen_seconds, "tool_calls": state.tool_seconds},
        )
        output.extra_fields.update(
            {
                "mls_task": state.task_name,
                "mls_model_name": state.model_name,
                "mls_steps": state.steps,
                "mls_tests": state.tests,
                "mls_done": state.done,
                "mls_reward": state.reward,
                "mls_finished_reason": state.finished_reason,
                "mls_max_steps": state.max_steps,
                "mls_max_tests": state.max_tests,
                "mls_budget_tier": state.budget_tier,
                "mls_budget_source": state.budget_source,
            }
        )
        return output

    async def _episode(self, state: _EpisodeState, sampling_params: dict[str, Any]) -> None:
        # 1) spawn per-episode environment worker (blocking -> executor)
        state.client, init = await self.loop.run_in_executor(
            None, self._spawn_worker, state.task_name, state.model_name,
            state.max_steps, state.max_tests,
        )
        state.log_dir = init.get("log_dir", "")
        tool_schemas = init["tool_schemas"]
        schemas_by_name = {s["name"]: s for s in tool_schemas}
        openai_tools = anthropic_to_openai_tools(tool_schemas)

        messages = [
            {"role": "system", "content": init["system_prompt"]},
            {"role": "user", "content": init["initial_prompt"]},
        ]
        state.prompt_ids = await self.apply_chat_template(messages, tools=openai_tools)
        prompt_len = len(state.prompt_ids)
        full_ids = list(state.prompt_ids)
        request_id = uuid.uuid4().hex

        while True:
            budget = self.response_length - len(state.response_mask)
            if budget <= 0:
                state.finished_reason = "response_budget"
                break
            sp = dict(sampling_params)
            sp["max_tokens"] = budget

            t0 = time.time()
            out = await self.server_manager.generate(
                request_id=request_id, prompt_ids=full_ids, sampling_params=sp
            )
            state.gen_seconds += time.time() - t0

            gen_ids = list(out.token_ids)
            full_ids += gen_ids
            state.response_ids = full_ids[prompt_len:]
            state.response_mask += [1] * len(gen_ids)
            state.assistant_turns += 1
            state.turn_log.append({"role": "assistant", "tokens": len(gen_ids)})

            if len(state.response_mask) >= self.response_length:
                state.finished_reason = "response_budget"
                break
            clean_stop = bool(gen_ids) and (gen_ids[-1] == self.im_end_id or gen_ids[-1] == self.eos_id)
            if not clean_stop:
                state.finished_reason = "generation_truncated"
                break

            text = await self.loop.run_in_executor(None, lambda ids=gen_ids: self.tokenizer.decode(ids))
            # Reasoning-parser step BEFORE tool parsing (vLLM chat-completions
            # order; verl #4757/#6424): only content after </think> may act.
            calls = parse_qwen_xml_tool_calls(strip_reasoning(text), schemas_by_name)

            if not calls:
                if state.nudges < self.nudge_limit and not state.done:
                    state.nudges += 1
                    delta = self._delta_ids(NUDGE_TEXT, wrap_tool_response=False)
                    if len(state.response_mask) + len(delta) >= self.response_length:
                        state.finished_reason = "response_budget"
                        break
                    full_ids += delta
                    state.response_ids = full_ids[prompt_len:]
                    state.response_mask += [0] * len(delta)
                    state.user_turns += 1
                    state.turn_log.append({"role": "nudge", "tokens": len(delta)})
                    continue
                state.finished_reason = "no_tool_call"
                break

            tool_name, tool_args = calls[0]
            t1 = time.time()
            dispatch = lambda: state.client.call(cmd="dispatch", name=tool_name, args=tool_args)  # noqa: E731
            if tool_name == "test":
                async with self._get_test_semaphore():
                    rsp = await self.loop.run_in_executor(None, dispatch)
            else:
                rsp = await self.loop.run_in_executor(None, dispatch)
            state.tool_seconds += time.time() - t1

            state.steps = rsp.get("step_count", state.steps)
            state.tests = rsp.get("test_count", state.tests)
            state.done = bool(rsp.get("done", False))
            result = str(rsp.get("result", ""))

            if state.done:
                state.finished_reason = "submitted"
                break
            if rsp.get("budget_exhausted") or state.steps >= state.max_steps:
                # per-episode action budget (extra_info.budget or env default);
                # the worker also HARD-rejects any dispatch past the budget.
                state.finished_reason = "step_budget"
                break

            delta = self._delta_ids(self._truncate_tool_response(result), wrap_tool_response=True)
            if len(state.response_mask) + len(delta) >= self.response_length:
                state.finished_reason = "response_budget"
                break
            full_ids += delta
            state.response_ids = full_ids[prompt_len:]
            state.response_mask += [0] * len(delta)
            state.user_turns += 1
            state.turn_log.append({"role": "tool", "tool": tool_name, "tokens": len(delta)})

    def _write_episode_log(self, state: _EpisodeState, elapsed: float) -> None:
        try:
            os.makedirs(self.episode_log_dir, exist_ok=True)
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "task": state.task_name,
                "model_name": state.model_name,
                "reward": state.reward,
                "score_detail": state.score_detail,
                "finished_reason": state.finished_reason,
                "timed_out": state.timed_out,
                "error": state.error,
                "elapsed_s": round(elapsed, 1),
                "gen_s": round(state.gen_seconds, 1),
                "tool_s": round(state.tool_seconds, 1),
                "assistant_turns": state.assistant_turns,
                "user_turns": state.user_turns,
                "nudges": state.nudges,
                "steps": state.steps,
                "tests": state.tests,
                "done": state.done,
                "max_steps": state.max_steps,
                "max_tests": state.max_tests,
                "budget_tier": state.budget_tier,
                "budget_source": state.budget_source,
                "prompt_tokens": len(state.prompt_ids),
                "response_tokens": len(state.response_ids),
                "assistant_token_sum": sum(state.response_mask),
                "mls_log_dir": state.log_dir,
                "turns": state.turn_log,
            }
            path = os.path.join(self.episode_log_dir, f"episodes_{os.getpid()}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass
