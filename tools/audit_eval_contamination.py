#!/usr/bin/env python3
"""Audit overlap between SFT training data and the eval datasets we track.

The audit is intentionally conservative:
  * FrontierCS/ALE: compare against the dumped eval prompts in experiments/raw_outputs.
  * MLS-Bench: compare against the 20 tasks that were actually evaluated, using the
    local MLS-Bench task_description.md files when available.
  * ThetaEvolve: only weak text evidence is available in this repo, so it is
    included as a caveated source from config/program files.

The main signal is exact token n-gram overlap after light normalization. A second
slug-level check catches same-task MLS contamination even when the training prompt
is a paraphrase of the benchmark task description.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLS_ROOT = "/srv/home/bohanlyu/MLS-Bench/tasks"

TRAIN_PATHS = [
    "sft/innovation_sft.jsonl.gz",
    "sft/innovation_wave2_sft.jsonl.gz",
    "sft/innovation_v4_sft.jsonl.gz",
]

TAG_PATHS = {
    "innovation_sft.jsonl.gz": "sft/_sft_tags.jsonl",
    "innovation_wave2_sft.jsonl.gz": "sft/_wave2_tags.jsonl",
    "innovation_v4_sft.jsonl.gz": "sft/_v4_tags.jsonl",
}

FRONTIER_PROMPTS = [
    "experiments/raw_outputs/frontiercs_algorithm/prompts.jsonl.gz",
    "experiments/raw_outputs/frontiercs_research_cpu/prompts.jsonl.gz",
    "experiments/raw_outputs/frontiercs_research_gpu/prompts.jsonl.gz",
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TOKEN_RE = re.compile(r"[a-z0-9_]+")
SPACE_RE = re.compile(r"\s+")


@dataclass
class EvalRecord:
    idx: int
    eval_id: str
    benchmark: str
    title: str
    source_path: str
    text: str
    tokens: List[str] = field(default_factory=list)


@dataclass
class TrainRecord:
    dataset: str
    path: str
    line_no: int
    tag_id: str
    tag_kind: str
    human_text: str
    assistant_text: str


@dataclass
class PairStat:
    matched_positions: set = field(default_factory=set)
    occurrences: int = 0
    longest_span: int = 0
    train_start: int = 0
    eval_start: int = 0
    scope: str = ""


def repo_path(path: str) -> str:
    return os.path.join(ROOT, path)


def open_text(path: str):
    full = repo_path(path) if not os.path.isabs(path) else path
    if full.endswith(".gz"):
        return gzip.open(full, "rt", encoding="utf-8", errors="replace")
    return open(full, "r", encoding="utf-8", errors="replace")


def read_text(path: str) -> str:
    with open_text(path) as f:
        return f.read()


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(strip_ansi(text).lower())


def snippet_from_tokens(tokens: List[str], start: int, span: int, limit: int = 44) -> str:
    if not tokens:
        return ""
    lo = max(0, start)
    hi = min(len(tokens), lo + min(span, limit))
    return " ".join(tokens[lo:hi])


def norm_slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def frontier_benchmark_from_path(path: str) -> str:
    parts = path.split("/")
    try:
        return parts[parts.index("raw_outputs") + 1]
    except (ValueError, IndexError):
        return os.path.basename(os.path.dirname(path))


def strip_frontier_prompt(prompt: str) -> str:
    text = prompt
    text = re.sub(
        r"^\s*You are a competitive programmer\..*?(?:\n\n|\r\n\r\n)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\n\s*Generate solution code:\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_title(text: str) -> str:
    for line in text.splitlines():
        clean = line.strip(" #\t")
        if not clean:
            continue
        if clean.lower() in {"problem", "description", "input", "output"}:
            continue
        if len(clean) > 160:
            clean = clean[:160]
        return clean
    return "untitled"


def load_frontier_records() -> List[EvalRecord]:
    out: List[EvalRecord] = []
    for path in FRONTIER_PROMPTS:
        full = repo_path(path)
        if not os.path.exists(full):
            continue
        benchmark = frontier_benchmark_from_path(path)
        with open_text(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = strip_frontier_prompt(row.get("prompt", ""))
                title = extract_title(text)
                data_source = row.get("data_source") or benchmark
                idx = row.get("problem_idx", len(out))
                eval_id = f"{benchmark}:{data_source}:idx{idx}:{norm_slug(title)[:80]}"
                out.append(
                    EvalRecord(
                        idx=len(out),
                        eval_id=eval_id,
                        benchmark=f"{benchmark}/{data_source}",
                        title=title,
                        source_path=path,
                        text=text,
                    )
                )
    return out


def mls_tasks_from_summary() -> List[str]:
    path = repo_path("experiments/raw_outputs/mlsbench/q35_inst_start/summary.json")
    if not os.path.exists(path):
        return []
    data = json.load(open(path, encoding="utf-8"))
    return [t["task"] for t in data.get("tasks", []) if t.get("task")]


def load_mls_records() -> List[EvalRecord]:
    out: List[EvalRecord] = []
    for task in mls_tasks_from_summary():
        candidates = [
            os.path.join(MLS_ROOT, task, "task_description.md"),
            repo_path(f"experiments/raw_outputs/mlsbench/q35_inst_start/task_logs/{task}.log.gz"),
        ]
        text = ""
        source = ""
        if os.path.exists(candidates[0]):
            source = candidates[0]
            text = read_text(source)
        elif os.path.exists(candidates[1]):
            source = candidates[1]
            text = extract_initial_prompt_from_log(read_text(source))
        if not text:
            continue
        title = extract_title(text)
        out.append(
            EvalRecord(
                idx=-1,
                eval_id=f"mlsbench:{task}",
                benchmark="mlsbench",
                title=title,
                source_path=source,
                text=text,
            )
        )
    return out


def extract_initial_prompt_from_log(text: str) -> str:
    text = strip_ansi(text)
    marker = "Initial prompt"
    i = text.find(marker)
    if i < 0:
        return ""
    text = text[i:]
    end_markers = ["[INFO]", "Step 1", "(total "]
    end = len(text)
    for marker in end_markers:
        j = text.find(marker)
        if j > 0:
            end = min(end, j)
    lines = []
    for line in text[:end].splitlines():
        line = line.strip()
        line = re.sub(r"^Initial prompt.*$", "", line)
        line = re.sub(r"^=+$", "", line)
        line = re.sub(r"^\(total .*$", "", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def load_theta_records() -> List[EvalRecord]:
    base = repo_path("experiments/raw_outputs/thetaevolve_circle_packing")
    if not os.path.isdir(base):
        return []
    parts = []
    for rel in [
        "q35_inst_start/config_used.yaml",
        "q35_inst_start/best_program.py",
    ]:
        path = os.path.join(base, rel)
        if os.path.exists(path):
            parts.append(read_text(path))
    if not parts:
        return []
    text = "\n\n".join(parts)
    return [
        EvalRecord(
            idx=-1,
            eval_id="thetaevolve:circle_packing_modular",
            benchmark="thetaevolve_circle_packing",
            title="circle_packing_modular",
            source_path="experiments/raw_outputs/thetaevolve_circle_packing/q35_inst_start",
            text=text,
        )
    ]


def load_eval_records(include_theta: bool = True) -> List[EvalRecord]:
    records = load_frontier_records() + load_mls_records()
    if include_theta:
        records.extend(load_theta_records())
    for i, rec in enumerate(records):
        rec.idx = i
        rec.tokens = tokenize(rec.text)
    return records


def load_tags(train_path: str) -> List[dict]:
    base = os.path.basename(train_path)
    tag_rel = TAG_PATHS.get(base)
    if not tag_rel:
        return []
    tag_path = repo_path(tag_rel)
    if not os.path.exists(tag_path):
        return []
    rows = []
    with open(tag_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_train_records(train_paths: Iterable[str]) -> Iterable[TrainRecord]:
    for rel in train_paths:
        full = repo_path(rel)
        if not os.path.exists(full):
            continue
        tags = load_tags(rel)
        dataset = os.path.basename(rel).replace(".jsonl.gz", "")
        with open_text(rel) as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                tag = tags[line_no - 1] if line_no - 1 < len(tags) else {}
                human_parts = []
                assistant_parts = []
                for turn in row.get("conversations", []):
                    value = turn.get("value") or ""
                    role = turn.get("from")
                    if role == "human":
                        human_parts.append(value)
                    elif role in {"gpt", "function_call", "observation"}:
                        assistant_parts.append(value)
                yield TrainRecord(
                    dataset=dataset,
                    path=rel,
                    line_no=line_no,
                    tag_id=str(tag.get("id") or ""),
                    tag_kind=str(tag.get("kind") or tag.get("domain") or ""),
                    human_text="\n\n".join(human_parts),
                    assistant_text="\n\n".join(assistant_parts),
                )


def build_eval_index(records: List[EvalRecord], ngram: int):
    token_to_id: Dict[str, int] = {}
    index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    powers = [1]
    base = 1_000_003
    mask = (1 << 64) - 1
    for _ in range(ngram):
        powers.append((powers[-1] * base) & mask)

    def token_id(tok: str) -> int:
        cur = token_to_id.get(tok)
        if cur is None:
            cur = len(token_to_id) + 1
            token_to_id[tok] = cur
        return cur

    for rec in records:
        ids = [token_id(t) for t in rec.tokens]
        if len(ids) < ngram:
            continue
        h = 0
        for val in ids[:ngram]:
            h = ((h * base) + val) & mask
        index[h].append((rec.idx, 0))
        high_pow = powers[ngram - 1]
        for pos in range(1, len(ids) - ngram + 1):
            h = (h - ((ids[pos - 1] * high_pow) & mask)) & mask
            h = ((h * base) + ids[pos + ngram - 1]) & mask
            index[h].append((rec.idx, pos))
    return token_to_id, index, base, mask, powers[ngram - 1]


def scan_scope(
    text: str,
    token_to_id: Dict[str, int],
    index: Dict[int, List[Tuple[int, int]]],
    base: int,
    mask: int,
    high_pow: int,
    ngram: int,
    train_key: Tuple[str, int, str],
    scope: str,
    stats: Dict[Tuple[Tuple[str, int, str], str, int], PairStat],
) -> Tuple[int, int]:
    toks = tokenize(text)
    if len(toks) < ngram:
        return len(toks), 0
    ids = [token_to_id.get(t, 0) for t in toks]
    zero_count = ids[:ngram].count(0)
    h = 0
    for val in ids[:ngram]:
        h = ((h * base) + val) & mask

    active_runs: Dict[Tuple[int, int], Tuple[int, int]] = {}
    matches = 0

    for pos in range(0, len(ids) - ngram + 1):
        if zero_count == 0:
            hits = index.get(h)
            if hits:
                matches += len(hits)
                for eval_idx, eval_pos in hits:
                    stat_key = (train_key, scope, eval_idx)
                    stat = stats.get(stat_key)
                    if stat is None:
                        stat = PairStat(scope=scope)
                        stats[stat_key] = stat
                    stat.occurrences += 1
                    stat.matched_positions.add(eval_pos)

                    diff = pos - eval_pos
                    run_key = (eval_idx, diff)
                    last_pos, run = active_runs.get(run_key, (-2, 0))
                    if last_pos == pos - 1:
                        run += 1
                    else:
                        run = 1
                    active_runs[run_key] = (pos, run)
                    span = ngram + run - 1
                    if span > stat.longest_span:
                        stat.longest_span = span
                        stat.train_start = pos - run + 1
                        stat.eval_start = eval_pos - run + 1
        if pos == len(ids) - ngram:
            break
        old = ids[pos]
        new = ids[pos + ngram]
        if old == 0:
            zero_count -= 1
        if new == 0:
            zero_count += 1
        h = (h - ((old * high_pow) & mask)) & mask
        h = ((h * base) + new) & mask
    return len(toks), matches


def severity(eval_cover: float, longest: int) -> str:
    if eval_cover >= 0.8 or longest >= 240:
        return "near_exact"
    if eval_cover >= 0.25 or longest >= 120:
        return "strong"
    if eval_cover >= 0.08 or longest >= 60:
        return "weak"
    return "ignore"


def audit_ngram_overlap(records: List[EvalRecord], train_paths: List[str], ngram: int, limit: int):
    token_to_id, index, base, mask, high_pow = build_eval_index(records, ngram)
    stats: Dict[Tuple[Tuple[str, int, str], str, int], PairStat] = {}
    train_meta: Dict[Tuple[str, int, str], TrainRecord] = {}
    totals = Counter()

    for rec in iter_train_records(train_paths):
        train_key = (rec.dataset, rec.line_no, rec.tag_id)
        train_meta[train_key] = rec
        totals["train_examples"] += 1
        totals[f"train_examples:{rec.dataset}"] += 1
        for scope, text in [("human", rec.human_text), ("assistant", rec.assistant_text)]:
            ntok, nhit = scan_scope(
                text,
                token_to_id,
                index,
                base,
                mask,
                high_pow,
                ngram,
                train_key,
                scope,
                stats,
            )
            totals[f"tokens:{scope}"] += ntok
            totals[f"hits:{scope}"] += nhit

    findings = []
    for (train_key, scope, eval_idx), stat in stats.items():
        ev = records[eval_idx]
        denom = max(1, len(ev.tokens) - ngram + 1)
        eval_cover = len(stat.matched_positions) / denom
        sev = severity(eval_cover, stat.longest_span)
        if sev == "ignore":
            continue
        tr = train_meta[train_key]
        train_tokens = tokenize(tr.human_text if scope == "human" else tr.assistant_text)
        findings.append(
            {
                "severity": sev,
                "eval_cover": round(eval_cover, 4),
                "matched_eval_ngrams": len(stat.matched_positions),
                "eval_ngrams": denom,
                "longest_span_tokens": stat.longest_span,
                "scope": scope,
                "train_dataset": tr.dataset,
                "train_path": tr.path,
                "train_line": tr.line_no,
                "train_id": tr.tag_id,
                "train_kind": tr.tag_kind,
                "eval_id": ev.eval_id,
                "eval_benchmark": ev.benchmark,
                "eval_title": ev.title,
                "eval_source": ev.source_path,
                "train_snippet": snippet_from_tokens(train_tokens, stat.train_start, stat.longest_span),
                "eval_snippet": snippet_from_tokens(ev.tokens, stat.eval_start, stat.longest_span),
            }
        )

    findings.sort(
        key=lambda x: (
            {"near_exact": 3, "strong": 2, "weak": 1}.get(x["severity"], 0),
            x["eval_cover"],
            x["longest_span_tokens"],
        ),
        reverse=True,
    )
    return findings[:limit], totals


def base_train_id(tag_id: str) -> str:
    tag_id = (tag_id or "").split("#", 1)[0]
    tag_id = re.sub(r"#r\d+$", "", tag_id)
    return tag_id


def audit_slug_overlap(train_paths: List[str], mls_tasks: List[str]) -> dict:
    task_set = set(mls_tasks)
    by_task = defaultdict(list)
    counts = Counter()
    for rec in iter_train_records(train_paths):
        tid = base_train_id(rec.tag_id)
        if tid in task_set:
            by_task[tid].append(
                {
                    "dataset": rec.dataset,
                    "path": rec.path,
                    "line": rec.line_no,
                    "id": rec.tag_id,
                    "kind": rec.tag_kind,
                }
            )
            counts[tid] += 1
    return {
        "matched_tasks": sorted(by_task),
        "n_matched_tasks": len(by_task),
        "n_eval_tasks": len(mls_tasks),
        "n_training_examples": sum(counts.values()),
        "by_task": {k: by_task[k] for k in sorted(by_task)},
    }


def summarize_eval(records: List[EvalRecord]) -> dict:
    by_benchmark = Counter(r.benchmark for r in records)
    return {
        "n_eval_records": len(records),
        "by_benchmark": dict(sorted(by_benchmark.items())),
        "records": [
            {
                "eval_id": r.eval_id,
                "benchmark": r.benchmark,
                "title": r.title,
                "source_path": r.source_path,
                "tokens": len(r.tokens),
            }
            for r in records
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--output", default="")
    ap.add_argument("--no-theta", action="store_true")
    ap.add_argument("train_paths", nargs="*", default=TRAIN_PATHS)
    args = ap.parse_args()

    records = load_eval_records(include_theta=not args.no_theta)
    findings, totals = audit_ngram_overlap(records, args.train_paths, args.ngram, args.limit)
    slug = audit_slug_overlap(args.train_paths, mls_tasks_from_summary())
    result = {
        "ngram": args.ngram,
        "train_paths": args.train_paths,
        "eval_summary": summarize_eval(records),
        "slug_overlap": slug,
        "scan_totals": dict(totals),
        "findings": findings,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(repo_path(args.output), "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
