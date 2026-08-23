"""nb_agent.py — in-container bounded agent for the NatureBench Apptainer harness.

Runs INSIDE the task container (apptainer exec). Talks to a local
OpenAI-compatible vLLM endpoint, writes /workspace/run.py, executes it, and
submits the output to the official host-side eval service (/evaluate).

Design goals (per project instructions): the AGENT side is a short, bounded
loop — not a 4h autonomous CLI session. The SCORING side is untouched: scoring
is whatever the official eval_service.py + task evaluator.py compute.

Required env:
    OPENAI_BASE_URL   e.g. http://127.0.0.1:8000/v1
    OPENAI_API_KEY    any non-empty string for local vLLM
    NB_MODEL          served model name
    EVAL_SERVICE_URL  e.g. http://127.0.0.1:8321
    TASK_NAME         e.g. s43588-024-00689-2
    BATCH_NAME        batch id used at /register time
    DATA_DIR          /task/problem/data
    OUTPUT_DIR        output dir (same absolute path inside and outside container)

Optional env:
    NB_MAX_ROUNDS       max LLM write/run/evaluate rounds (default 4)
    NB_RUN_TIMEOUT_S    per-execution timeout for run.py (default 1200)
    NB_AGENT_BUDGET_S   total wall budget for the agent loop (default 2400)
    NB_MAX_TOKENS       completion cap per LLM call (default 8192)
    NB_TEMPERATURE      sampling temperature (default 0.2)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Prompt scaffold mirrors repo/agent/base.py SYSTEM_PROMPT (base mode) so the
# task framing matches the official single-shot agents.
SYSTEM_PROMPT = """# Role
You are an expert AI Researcher and Scientific Coder. Your goal is to solve the Scientific Machine Learning task described below.

# Environment
Your `run.py` will be executed with two environment variables:
- `DATA_DIR`: path to the read-only data directory (contains instance sub-folders).
- `OUTPUT_DIR`: path where you must write output files (one sub-folder per instance).

# Rules
1.  **Interface Compliance**: Implement `run.py` exactly matching the template in README Section 6. Use `DATA_DIR` and `OUTPUT_DIR` environment variables.
2.  **Output Format**: Save results to `output/{{instance_name}}/` in the exact format specified in README Section 5 (file name, shape, dtype, value range).
3.  **Completeness Over Brevity**: Do NOT simplify the solution logic or skip steps just to keep the file short. Implement a fully functional, robust solution.
4.  **Metric Optimization**: Optimize for the Metrics (especially Primary Metric) in README.
5.  **Production Ready**: Code should be clean and handle missing values / edge cases from the Data Description.
6.  **Read-Only Data**: Do NOT modify any files in the data directory. Only write to the output directory.
7.  **Anti-Cheating & Data Integrity**: You are strictly FORBIDDEN from using ground truth labels (targets) corresponding to the test set for training, validation, feature engineering, or directly as output values.
8.  **Self-Contained**: run.py must run start-to-finish without user input and finish within the per-run time limit. Prefer fast, CPU-efficient methods unless a GPU is explicitly available.

# Output Format
1. **Research Plan** (3-5 sentences): Goal, Method, Reasoning
2. **Implementation**: Single ```python``` code block with the complete `run.py`
"""

FEEDBACK_TEMPLATE = """## Round {round} feedback

Your previous `run.py` {outcome}.

```
{detail}
```

{instruction}
Reply with the SAME format: a short plan, then ONE ```python``` code block with the COMPLETE corrected/improved run.py (not a diff).
"""


def log(msg: str) -> None:
    print(f"[nb_agent {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def chat_completion(base_url: str, api_key: str, model: str, messages,
                    max_tokens: int, temperature: float, timeout: float = 600.0):
    """Stdlib-only OpenAI chat.completions call (no external deps in container)."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"] or ""


def post_json(url: str, payload: dict, timeout: float = 3600.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"error": str(e)}


def extract_code(text: str):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        return None
    # longest block is the most likely complete script
    return max(blocks, key=len).strip()


def read_task_docs(problem_dir: Path, per_file_cap: int = 9000) -> str:
    parts = []
    for name in ("README.md", "data_description.md"):
        p = problem_dir / name
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="replace")
            if len(txt) > per_file_cap:
                txt = txt[:per_file_cap] + "\n\n[... truncated for context length ...]"
            parts.append(f"---\n**{name}**\n\n{txt.strip()}")
    return "\n\n".join(parts)


def main() -> int:
    base_url = os.environ["OPENAI_BASE_URL"]
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    model = os.environ["NB_MODEL"]
    eval_url = os.environ["EVAL_SERVICE_URL"].rstrip("/")
    task_name = os.environ["TASK_NAME"]
    batch_name = os.environ.get("BATCH_NAME", "default")
    data_dir = os.environ["DATA_DIR"]
    output_dir = os.environ["OUTPUT_DIR"]
    max_rounds = int(os.environ.get("NB_MAX_ROUNDS", "4"))
    run_timeout = int(os.environ.get("NB_RUN_TIMEOUT_S", "1200"))
    budget = int(os.environ.get("NB_AGENT_BUDGET_S", "2400"))
    max_tokens = int(os.environ.get("NB_MAX_TOKENS", "8192"))
    temperature = float(os.environ.get("NB_TEMPERATURE", "0.2"))

    workspace = Path("/workspace")
    problem_dir = Path("/task/problem")
    (workspace / "output").mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    docs = read_task_docs(problem_dir)
    data_listing = subprocess.run(
        ["bash", "-c", f"find {data_dir} -maxdepth 2 | head -80"],
        capture_output=True, text=True,
    ).stdout

    user_msg = (
        SYSTEM_PROMPT
        + "\n\n# Task Documentation\n\n" + docs
        + "\n\n# Data Directory Listing\n\n```\n" + data_listing + "\n```"
        + f"\n\n# Instances\nThe data directory is `{data_dir}`; write outputs under `{output_dir}` "
          "via the DATA_DIR / OUTPUT_DIR environment variables (they are set for you)."
    )

    messages = [{"role": "user", "content": user_msg}]
    t0 = time.time()
    best = None
    transcript = []

    for rnd in range(1, max_rounds + 1):
        if time.time() - t0 > budget:
            log(f"budget exhausted before round {rnd}")
            break
        # Keep the context bounded: the task brief (messages[0]) plus the most
        # recent attempt + its feedback. Older failed attempts add tokens
        # without adding information, and would overflow the served context.
        call_messages = messages if len(messages) <= 3 else [messages[0]] + messages[-2:]
        log(f"round {rnd}/{max_rounds}: calling model ({model}), "
            f"{len(call_messages)} msgs")
        try:
            reply = chat_completion(base_url, api_key, model, call_messages,
                                    max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            log(f"LLM call failed: {e}")
            if len(call_messages) > 1:   # retry once with just the task brief
                try:
                    reply = chat_completion(base_url, api_key, model, [messages[0]],
                                            max_tokens=max_tokens, temperature=temperature)
                    log("recovered with task-brief-only context")
                except Exception as e2:
                    log(f"retry also failed: {e2}")
                    break
            else:
                break
        transcript.append({"round": rnd, "reply_chars": len(reply)})

        code = extract_code(reply)
        if not code:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                             "No python code block found. Reply with ONE ```python``` block containing the complete run.py."})
            continue

        run_py = workspace / "run.py"
        run_py.write_text(code, encoding="utf-8")
        log(f"wrote {run_py} ({len(code)} chars); executing (timeout={run_timeout}s)")

        env = dict(os.environ)
        env["DATA_DIR"] = data_dir
        env["OUTPUT_DIR"] = output_dir
        try:
            proc = subprocess.run(
                [sys.executable, str(run_py)],
                cwd=str(workspace), env=env,
                capture_output=True, text=True, timeout=run_timeout,
            )
            run_rc = proc.returncode
            run_out = (proc.stdout or "")[-3000:]
            run_err = (proc.stderr or "")[-3000:]
        except subprocess.TimeoutExpired:
            run_rc = -9
            run_out, run_err = "", f"run.py exceeded {run_timeout}s and was killed"
        log(f"run.py exit={run_rc}")

        if run_rc == 0:
            log("run.py succeeded; submitting to official /evaluate")
            status, resp = post_json(f"{eval_url}/evaluate", {
                "task_name": task_name,
                "batch_name": batch_name,
                "output_dir": output_dir,
            })
            agg = resp.get("aggregate_improvement")
            log(f"/evaluate status={status} aggregate_improvement={agg} "
                f"best={resp.get('best_aggregate_improvement')}")
            transcript[-1]["evaluate"] = resp
            if isinstance(agg, (int, float)):
                if best is None or agg > best:
                    best = agg
                outcome = f"ran successfully. Official score for this attempt: aggregate_improvement = {agg:.6f} (best so far: {resp.get('best_aggregate_improvement')})"
                detail = json.dumps(resp.get("per_instance_improvement", {}), indent=1)[:2000]
                instruction = ("Now try to IMPROVE the score with a better method or better hyperparameters. "
                               "Higher aggregate_improvement is better; 0 means exactly matching SOTA, positive means surpassing SOTA.")
            else:
                outcome = "ran but the evaluator did not return a valid score"
                detail = json.dumps(resp)[:2500]
                instruction = "Fix run.py so that outputs validate correctly against the output-format spec."
        else:
            outcome = f"failed with exit code {run_rc}"
            detail = f"STDOUT (tail):\n{run_out}\n\nSTDERR (tail):\n{run_err}"
            instruction = "Fix the bug and return the complete corrected run.py."

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": FEEDBACK_TEMPLATE.format(
            round=rnd, outcome=outcome, detail=detail, instruction=instruction)})

    # Final best-score query (official protocol endpoint)
    try:
        url = f"{eval_url}/best_score?task_name={task_name}&batch_name={batch_name}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            best_resp = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        best_resp = {"error": str(e)}

    summary = {
        "task_name": task_name,
        "batch_name": batch_name,
        "model": model,
        "rounds_used": len(transcript),
        "harness_best_seen": best,
        "official_best_score_response": best_resp,
        "wall_seconds": round(time.time() - t0, 1),
        "transcript_meta": transcript,
    }
    (workspace / "agent_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"SUMMARY {json.dumps(summary['official_best_score_response'])[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
