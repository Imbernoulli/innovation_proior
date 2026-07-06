#!/usr/bin/env python3
"""Generate the data the static site needs for the Agentic and Training-data views.

Two outputs, both committed (small index + gzipped shards):

  1. agentic.json
       Index for the "Agentic" browse mode. One entry per task that has a
       trajectories/<task>/agentic_messages.json transcript, joined with the
       title/domain/year from trajectories.json. The transcripts themselves are
       fetched directly from trajectories/<task>/agentic_messages.json (already
       in the repo) — no duplication here.

  2. sft/viewer/index.json  +  sft/viewer/<dataset>-NNN.json.gz
       A browsable catalogue of EVERY SFT training example. index.json holds
       light per-example metadata (which dataset/shard, derived kind, title,
       year, turn/action counts, loss-masking + enable_thinking flags) plus
       dataset summaries. The full example bodies live in gzipped shards that
       the site lazy-loads (and gunzips in-browser) only when an example is
       opened, so the whole 200 MB+ corpus stays browsable without loading it
       all at once.

Run from the repo root:  python3 tools/build_site_data.py
"""
import gzip
import io
import json
import os
import re
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from categorize import category_for  # noqa: E402  (canonical coarse-category map)

# ----------------------------------------------------------------------------
# 1. Agentic index
# ----------------------------------------------------------------------------
def count_messages(rec):
    """(#user turns, #assistant turns, #tool-calls, #tool results) for a transcript."""
    msgs = rec.get("messages", [])
    users = sum(1 for m in msgs if m.get("role") == "user")
    asst = sum(1 for m in msgs if m.get("role") == "assistant")
    tools = sum(1 for m in msgs if m.get("role") == "tool")
    calls = sum(len(m.get("tool_calls") or []) for m in msgs if m.get("role") == "assistant")
    return users, asst, tools, calls


def build_agentic_index():
    traj = json.load(open(os.path.join(ROOT, "trajectories.json")))
    meta_by_task = {t["task"]: t for t in traj}

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "trajectories", "*", "agentic_messages.json"))):
        task = os.path.basename(os.path.dirname(path))
        try:
            rec = json.load(open(path))
        except Exception as e:
            print(f"  ! skip {task}: {e}")
            continue
        users, asst, tools, calls = count_messages(rec)
        m = meta_by_task.get(task, {})
        rows.append({
            "task": task,
            "title": m.get("title") or task,
            "domain": m.get("domain") or "Other",
            "category": m.get("category") or category_for(m.get("domain")),
            "year": m.get("year"),
            "endpoint": m.get("endpoint") or "",
            "n_tools": len(rec.get("tools") or []),
            "n_steps": asst,        # assistant turns = reasoning+action steps
            "n_actions": calls,     # individual tool calls
        })

    rows.sort(key=lambda r: (r["domain"].lower(), r["title"].lower()))
    out = os.path.join(ROOT, "agentic.json")
    with open(out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"agentic.json: {len(rows)} tasks -> {out}")


# ----------------------------------------------------------------------------
# 2. Training-data viewer (index + gzipped shards)
# ----------------------------------------------------------------------------
SHARD_BUDGET = 3_000_000          # ~uncompressed bytes per shard before flush
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOLCALL = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_HEAD = re.compile(r"^\s*#+\s*")


def first_human(convs):
    for c in convs:
        if c.get("from") == "human":
            return c.get("value", "")
    return convs[0].get("value", "") if convs else ""


def derive_title(convs):
    """A short, human-readable label: the first descriptive prose line of the
    first user turn (skipping markdown headings, fenced blocks, tags, and bare
    path/identifier tokens that aren't real titles)."""
    txt = first_human(convs)
    for raw in txt.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):           # markdown heading = section name, not a title
            continue
        if stripped.startswith(("<", "```", "|", "---", "===", "*", "-", ">")):
            continue
        if " " not in stripped:                # bare path / identifier / single token
            continue
        if len(stripped) < 12:
            continue
        line = re.sub(r"\s+", " ", stripped)
        return line[:96] + ("…" if len(line) > 96 else "")
    # fall back to the very first non-empty line if nothing prose-like was found
    for raw in txt.splitlines():
        s = _HEAD.sub("", raw).strip()
        if s:
            s = re.sub(r"\s+", " ", s)
            return s[:96] + ("…" if len(s) > 96 else "")
    return "(untitled)"


def derive_year(ex):
    sysmsg = ex.get("system") or ""
    m = _YEAR.search(sysmsg)
    return int(m.group(1)) if m else None


def classify(ex, dataset):
    """Return (group, kind, flags) describing an example for the catalogue."""
    convs = ex.get("conversations", [])
    roles = [c.get("from") for c in convs]
    has_fc = "function_call" in roles
    has_obs = "observation" in roles
    loss_flags = [c.get("loss") for c in convs if "loss" in c]
    masked = any(v is False for v in loss_flags)
    et = ex.get("enable_thinking")
    thinking = et is not False  # None/absent => thinking template default (True)
    n_user = roles.count("human")
    n_fc = roles.count("function_call")
    n_obs = roles.count("observation")
    multi = len(convs) > 2

    if dataset == "innovation_sft":
        if has_fc:
            kind = "agentic · folded" if masked else "agentic · full"
            group = "agentic"
        elif multi:
            kind = "trajectory · folded" if masked else "trajectory · full"
            group = "trajectory"
        else:
            kind = "method (single-turn)"
            group = "method"
    else:  # maintain_sft
        if et is False:
            kind = "Open-SWE Qwen3.5 · non-thinking"
            group = "maintain · non-thinking"
        elif has_fc:
            kind = "agentic distill (tool-use)"
            group = "maintain · agentic"
        else:
            kind = "reasoning distill"
            group = "maintain · reasoning"

    flags = {
        "turns": n_user,
        "actions": n_fc,
        "obs": n_obs,
        "convs": len(convs),
        "thinking": thinking,
        "folded": bool(loss_flags),
        "masked": masked,
        "has_tools": bool(ex.get("tools")),
    }
    return group, kind, flags


def gzip_bytes(text):
    buf = io.BytesIO()
    # mtime=0 for reproducible output (no Date dependency, stable git blobs)
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def build_viewer():
    out_dir = os.path.join(ROOT, "sft", "viewer")
    os.makedirs(out_dir, exist_ok=True)
    # clean stale shards so removed examples don't linger
    for old in glob.glob(os.path.join(out_dir, "*.json.gz")):
        os.remove(old)

    datasets = [
        ("innovation_sft", os.path.join(ROOT, "sft", "innovation_sft.jsonl.gz")),
        # maintain_sft (HF-scraped capability-maintenance set) dropped 2026-07 — innovation-only now.
    ]

    examples_index = []
    summaries = {}
    gid = 0

    for ds, path in datasets:
        if not os.path.exists(path):
            print(f"  ! missing {path} — skipping {ds}")
            continue
        shard_no = 0
        buf_items = []
        buf_bytes = 0
        ds_count = 0
        by_kind = {}
        by_group = {}

        def flush():
            nonlocal shard_no, buf_items, buf_bytes
            if not buf_items:
                return
            name = f"{ds}-{shard_no:03d}.json.gz"
            payload = json.dumps(buf_items, ensure_ascii=False)
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(gzip_bytes(payload))
            shard_no += 1
            buf_items = []
            buf_bytes = 0

        for line in gzip.open(path, "rt", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            group, kind, flags = classify(ex, ds)
            shard_name = f"{ds}-{shard_no:03d}.json.gz"
            examples_index.append({
                "id": gid,
                "ds": ds,
                "shard": shard_name,
                "i": len(buf_items),
                "group": group,
                "kind": kind,
                "title": derive_title(ex.get("conversations", [])),
                "year": derive_year(ex),
                **flags,
            })
            buf_items.append(ex)
            buf_bytes += len(line)
            gid += 1
            ds_count += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1
            by_group[group] = by_group.get(group, 0) + 1
            if buf_bytes >= SHARD_BUDGET:
                flush()
        flush()

        summaries[ds] = {
            "count": ds_count,
            "shards": shard_no,
            "by_kind": dict(sorted(by_kind.items())),
            "by_group": dict(sorted(by_group.items())),
        }
        print(f"{ds}: {ds_count} examples in {shard_no} shards")

    index = {
        "total": len(examples_index),
        "datasets": summaries,
        "examples": examples_index,
    }
    out = os.path.join(out_dir, "index.json")
    with open(out, "w") as f:
        json.dump(index, f, ensure_ascii=False)
    sz = os.path.getsize(out)
    print(f"sft/viewer/index.json: {len(examples_index)} examples, {sz/1024:.0f} KB")


if __name__ == "__main__":
    build_agentic_index()
    build_viewer()
