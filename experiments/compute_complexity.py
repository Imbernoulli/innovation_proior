#!/usr/bin/env python3
"""Compute code-complexity metrics from raw FrontierSmith eval outputs."""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import statistics
import textwrap
import tokenize
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Iterable

try:
    from radon.complexity import cc_visit

    RADON_AVAILABLE = True
except Exception:
    cc_visit = None
    RADON_AVAILABLE = False


ROOT = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs"
OUT_DIR = "/scratch/gpfs/CHIJ/bohan/fs/innovation_prior/experiments"
REPORT_PATH = os.path.join(OUT_DIR, "complexity_codex.md")

ADVANCED_KEYWORDS = [
    "cross_fit",
    "propensity",
    "doubly_robust",
    "isotonic",
    "hnsw",
    "efSearch",
    "anneal",
    "beam_search",
    "kde",
    "orthogonal",
    "residuali",
    "meek",
    "ensemble",
    "batchnorm",
    "cosineanneal",
    "gradient_boost",
    "xgboost",
    "lightgbm",
    "catboost",
    "lasso",
    "ridge",
    "elastic_net",
    "random_forest",
    "adaboost",
    "automl",
    "optuna",
    "hyperopt",
    "bayesian_optim",
    "monte_carlo",
    "mcts",
    "simulated_anneal",
    "tabu_search",
    "aho_corasick",
    "suffix_array",
    "segment_tree",
    "fenwick",
    "treap",
    "splay",
    "kd_tree",
    "ball_tree",
    "lsh",
    "spectral",
    "svd",
    "pca",
    "nmf",
    "tsne",
    "umap",
    "dbscan",
    "isolation_forest",
    "viterbi",
    "crf",
    "hmm",
    "attention",
    "transformer",
    "bilstm",
    "gru",
    "conv1d",
    "conv2d",
    "wavelet",
    "fft",
    "kalman",
    "particle_filter",
    "em_algorithm",
    "gibbs",
    "mcmc",
    "importance_sampling",
]

METRIC_KEYS = [
    "loc",
    "ast_nodes",
    "cyclomatic",
    "imports",
    "fn_calls",
    "technique_kws",
]


@dataclass
class SolutionMetrics:
    loc: float
    ast_nodes: float
    cyclomatic: float
    imports: float
    fn_calls: float
    technique_kws: float


@dataclass
class Cell:
    benchmark: str
    group: str
    model: str
    source: str
    language: str
    attempted: int = 0
    parsed: int = 0
    empty_code: int = 0
    parse_errors: int = 0
    no_code_block: int = 0
    json_errors: int = 0
    scores_attempted: list[float] = field(default_factory=list)
    scores_parsed: list[float] = field(default_factory=list)
    metrics: list[SolutionMetrics] = field(default_factory=list)


@dataclass
class ResolvedSource:
    model: str
    group: str
    kind: str
    path: str | None
    candidates: list[str]
    benchmarks: list[str]
    language_by_benchmark: dict[str, str]


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def extract_fenced_code(text: str, language: str) -> str:
    text = strip_think(text or "")
    if language == "cpp":
        wanted = {"cpp", "c++", "cc", "cxx"}
    else:
        wanted = {"python", "py"}
    blocks: list[str] = []
    pattern = re.compile(r"```\s*([A-Za-z0-9_+#.-]*)[^\n`]*\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(text):
        lang = (match.group(1) or "").strip().lower()
        if lang in wanted:
            blocks.append(match.group(2))
    return blocks[-1].strip("\n") if blocks else ""


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def extract_mls_added_code(path: str) -> str:
    add_re = re.compile(r"^\+\s+\d+\s+\|\s?(.*)$")
    lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = strip_ansi(raw.rstrip("\n"))
            match = add_re.match(line)
            if match:
                lines.append(match.group(1).rstrip())
    return "\n".join(lines).strip("\n")


def normalize_python(code: str) -> str:
    return textwrap.dedent(code.strip("\n"))


def python_loc(code: str) -> int:
    code = normalize_python(code)
    if not code.strip():
        return 0
    comment_lines: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                comment_lines.add(tok.start[0])
    except tokenize.TokenError:
        pass
    count = 0
    for lineno, line in enumerate(code.splitlines(), start=1):
        if not line.strip():
            continue
        if lineno in comment_lines and line.lstrip().startswith("#"):
            continue
        count += 1
    return count


def cpp_loc(code: str) -> int:
    count = 0
    in_block = False
    for raw in code.splitlines():
        line = raw
        out: list[str] = []
        i = 0
        while i < len(line):
            if in_block:
                end = line.find("*/", i)
                if end == -1:
                    i = len(line)
                else:
                    in_block = False
                    i = end + 2
            elif line.startswith("/*", i):
                in_block = True
                i += 2
            elif line.startswith("//", i):
                break
            else:
                out.append(line[i])
                i += 1
        if "".join(out).strip():
            count += 1
    return count


def python_imports(tree: ast.AST) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                imports.add(f"{module}.{alias.name}".strip("."))
    return imports


def python_call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def python_fn_calls(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = python_call_name(node.func)
            if name:
                names.add(name)
    return names


def python_manual_cyclomatic(tree: ast.AST) -> int:
    complexity = 1
    decision_nodes = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.IfExp,
        ast.ExceptHandler,
        ast.comprehension,
    )
    for node in ast.walk(tree):
        if isinstance(node, decision_nodes):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += max(0, len(node.values) - 1)
    return complexity


def python_radon_cyclomatic(code: str) -> int:
    if not RADON_AVAILABLE or cc_visit is None:
        raise RuntimeError("radon is not available")
    blocks = cc_visit(code)
    if not blocks:
        return 1
    return int(sum(block.complexity for block in blocks))


CPP_CONTROL_WORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "sizeof",
    "return",
    "delete",
    "new",
    "static_cast",
    "dynamic_cast",
    "reinterpret_cast",
    "const_cast",
}


def cpp_imports(code: str) -> set[str]:
    includes: set[str] = set()
    for match in re.finditer(r"^\s*#\s*include\s*([<\"][^>\"]+[>\"])", code, re.MULTILINE):
        includes.add(match.group(1))
    return includes


def cpp_brace_pairs(code: str) -> int:
    depth = 0
    pairs = 0
    for ch in code:
        if ch == "{":
            depth += 1
        elif ch == "}" and depth > 0:
            pairs += 1
            depth -= 1
    return pairs


def cpp_cyclomatic(code: str) -> int:
    complexity = 1
    complexity += len(re.findall(r"\b(if|for|while)\b", code))
    complexity += code.count("&&")
    complexity += code.count("||")
    return complexity


def cpp_fn_calls(code: str) -> set[str]:
    names: set[str] = set()
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code):
        name = match.group(1)
        if name in CPP_CONTROL_WORDS:
            continue
        names.add(name)
    return names


def technique_hits(code: str) -> set[str]:
    lowered = code.lower()
    return {kw for kw in ADVANCED_KEYWORDS if kw.lower() in lowered}


def analyze_solution(code: str, language: str) -> SolutionMetrics:
    if language == "cpp":
        return SolutionMetrics(
            loc=cpp_loc(code),
            ast_nodes=cpp_brace_pairs(code),
            cyclomatic=cpp_cyclomatic(code),
            imports=len(cpp_imports(code)),
            fn_calls=len(cpp_fn_calls(code)),
            technique_kws=len(technique_hits(code)),
        )

    code = normalize_python(code)
    tree = ast.parse(code)
    if RADON_AVAILABLE:
        cyclomatic = python_radon_cyclomatic(code)
    else:
        cyclomatic = python_manual_cyclomatic(tree)
    return SolutionMetrics(
        loc=python_loc(code),
        ast_nodes=sum(1 for _ in ast.walk(tree)),
        cyclomatic=cyclomatic,
        imports=len(python_imports(tree)),
        fn_calls=len(python_fn_calls(tree)),
        technique_kws=len(technique_hits(code)),
    )


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def std(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return statistics.stdev(vals)


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{fmt_num(mean(values))} +/- {fmt_num(std(values))}"


def metric_values(cell: Cell, key: str) -> list[float]:
    return [float(getattr(item, key)) for item in cell.metrics]


def cell_mean(cell: Cell, key: str) -> float | None:
    return mean(metric_values(cell, key))


def rel(path: str | None) -> str:
    if path is None:
        return "MISSING"
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def first_existing(candidates: list[str]) -> str | None:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def fcsale_candidates(tag: str) -> list[str]:
    variants = [tag]
    if tag.startswith("r3_"):
        short = tag[len("r3_") :]
        variants.extend([short, f"r3_soup_{short}"])
    candidates: list[str] = []
    for variant in variants:
        candidates.append(
            os.path.join(
                ROOT,
                f"cc_eval_{variant}_thinking_32k_both_vllm",
                "shard_0",
                "samples.jsonl",
            )
        )
    if tag.startswith("r3_"):
        candidates.append(
            os.path.join(ROOT, f"cc_eval_all_r3_{tag[len('r3_'):]}", "fcsale", "shard_0", "samples.jsonl")
        )
    else:
        candidates.append(os.path.join(ROOT, f"cc_eval_all_r3_{tag}", "fcsale", "shard_0", "samples.jsonl"))
    # Prefer the all-in-one run when present for the exact requested r3 a10 campaign.
    if tag.startswith("r3_"):
        all_path = os.path.join(
            ROOT, f"cc_eval_all_r3_{tag[len('r3_'):]}", "fcsale", "shard_0", "samples.jsonl"
        )
        candidates = [all_path] + [c for c in candidates if c != all_path]
    return list(dict.fromkeys(candidates))


def research_candidates(tag: str) -> list[str]:
    variants = [tag]
    if tag.startswith("r3_"):
        variants.append(tag[len("r3_") :])
    candidates: list[str] = []
    for variant in variants:
        candidates.append(
            os.path.join(
                ROOT,
                f"cc_eval_{variant}_research_thinking_32k_vllm",
                "shard_0",
                "samples.jsonl",
            )
        )
    return list(dict.fromkeys(candidates))


def mls_candidates(tag: str) -> list[str]:
    candidates: list[str] = []
    if tag.startswith("r3_"):
        candidates.append(os.path.join(ROOT, f"cc_eval_all_r3_{tag[len('r3_'):]}", "mls", "task_logs"))
    candidates.append(os.path.join(ROOT, f"cc_eval_all_r3_{tag}", "mls", "task_logs"))
    candidates.append(os.path.join(ROOT, f"cc_mlsbench_cpu_{tag}", "task_logs"))
    return list(dict.fromkeys(candidates))


def resolve_sources() -> list[ResolvedSource]:
    sources: list[ResolvedSource] = []

    for tag in ["r3_methodtraj_v4_r3_a10", "r3_methodv4_r3_a10"]:
        candidates = fcsale_candidates(tag)
        sources.append(
            ResolvedSource(
                model=tag,
                group="OURS",
                kind="samples",
                path=first_existing(candidates),
                candidates=candidates,
                benchmarks=["FCS", "ALE"],
                language_by_benchmark={"FCS": "cpp", "ALE": "python"},
            )
        )

    for tag in ["retest_start", "clean_start"]:
        candidates = fcsale_candidates(tag)
        sources.append(
            ResolvedSource(
                model=tag,
                group="BASE",
                kind="samples",
                path=first_existing(candidates),
                candidates=candidates,
                benchmarks=["FCS", "ALE"],
                language_by_benchmark={"FCS": "cpp", "ALE": "python"},
            )
        )

    for tag, group in [("r3_methodv4_r3_a20", "OURS"), ("retest_start", "BASE")]:
        candidates = research_candidates(tag)
        sources.append(
            ResolvedSource(
                model=tag,
                group=group,
                kind="samples",
                path=first_existing(candidates),
                candidates=candidates,
                benchmarks=["Research"],
                language_by_benchmark={"Research": "python"},
            )
        )

    for tag, group in [
        ("r3_methodtraj_v4_r3_a10", "OURS"),
        ("r3_methodv4_r3_a10", "OURS"),
        ("q35_start_devfix", "BASE"),
    ]:
        candidates = mls_candidates(tag)
        sources.append(
            ResolvedSource(
                model=tag,
                group=group,
                kind="mls_logs",
                path=first_existing(candidates),
                candidates=candidates,
                benchmarks=["MLS"],
                language_by_benchmark={"MLS": "python"},
            )
        )

    return sources


def score_from_obj(obj: dict[str, Any]) -> float | None:
    score = (obj.get("metrics") or {}).get("score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def benchmark_from_data_source(data_source: str | None) -> str | None:
    ds = (data_source or "").lower()
    if ds in {"frontiercs", "fcs"}:
        return "FCS"
    if ds in {"alebench", "ale"}:
        return "ALE"
    if "research" in ds:
        return "Research"
    return None


def make_cell(source: ResolvedSource, benchmark: str) -> Cell:
    return Cell(
        benchmark=benchmark,
        group=source.group,
        model=source.model,
        source=source.path or "MISSING",
        language=source.language_by_benchmark[benchmark],
    )


def process_samples(source: ResolvedSource, cells: dict[tuple[str, str], Cell]) -> None:
    assert source.path is not None
    with open(source.path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                for benchmark in source.benchmarks:
                    cells[(source.model, benchmark)].json_errors += 1
                continue
            benchmark = benchmark_from_data_source(obj.get("data_source"))
            if benchmark not in source.benchmarks:
                continue
            cell = cells[(source.model, benchmark)]
            cell.attempted += 1
            score = score_from_obj(obj)
            if score is not None:
                cell.scores_attempted.append(score)
            code = extract_fenced_code(obj.get("text") or "", cell.language)
            if not code.strip():
                cell.empty_code += 1
                cell.no_code_block += 1
                continue
            try:
                metrics = analyze_solution(code, cell.language)
            except SyntaxError:
                cell.parse_errors += 1
                continue
            except Exception:
                cell.parse_errors += 1
                continue
            cell.metrics.append(metrics)
            cell.parsed += 1
            if score is not None:
                cell.scores_parsed.append(score)


def process_mls_logs(source: ResolvedSource, cells: dict[tuple[str, str], Cell]) -> None:
    assert source.path is not None
    cell = cells[(source.model, "MLS")]
    paths = sorted(glob.glob(os.path.join(source.path, "*.log")))
    for path in paths:
        cell.attempted += 1
        code = extract_mls_added_code(path)
        if not code.strip():
            cell.empty_code += 1
            continue
        try:
            metrics = analyze_solution(code, "python")
        except SyntaxError:
            cell.parse_errors += 1
            continue
        except Exception:
            cell.parse_errors += 1
            continue
        cell.metrics.append(metrics)
        cell.parsed += 1


def compute_cells(sources: list[ResolvedSource]) -> dict[tuple[str, str], Cell]:
    cells: dict[tuple[str, str], Cell] = {}
    for source in sources:
        for benchmark in source.benchmarks:
            cells[(source.model, benchmark)] = make_cell(source, benchmark)

    for source in sources:
        if source.path is None:
            continue
        if source.kind == "samples":
            process_samples(source, cells)
        elif source.kind == "mls_logs":
            process_mls_logs(source, cells)
    return cells


def table_for_benchmark(cells: dict[tuple[str, str], Cell], benchmark: str) -> str:
    rows = [
        "| model | LOC | AST_nodes | cyclomatic | imports | fn_calls | technique_kws | score | N_parsed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    bench_cells = [cell for (_model, bench), cell in cells.items() if bench == benchmark]
    group_order = {"OURS": 0, "BASE": 1}
    bench_cells.sort(key=lambda c: (group_order.get(c.group, 9), c.model))
    for cell in bench_cells:
        model = f"{cell.group}:{cell.model}"
        values = [fmt_mean_std(metric_values(cell, key)) for key in METRIC_KEYS]
        score = fmt_num(mean(cell.scores_attempted), 3)
        n_text = f"{cell.parsed}/{cell.attempted}"
        if cell.parsed < 5:
            n_text += " LOW_N"
        rows.append("| " + " | ".join([model] + values + [score, n_text]) + " |")
    return "\n".join(rows)


def pooled_group_metrics(cells: dict[tuple[str, str], Cell], benchmark: str, group: str) -> dict[str, Any]:
    selected = [cell for (_model, bench), cell in cells.items() if bench == benchmark and cell.group == group]
    out: dict[str, Any] = {
        "tags": [cell.model for cell in selected],
        "parsed": sum(cell.parsed for cell in selected),
        "attempted": sum(cell.attempted for cell in selected),
        "score_values": [],
    }
    for key in METRIC_KEYS:
        vals: list[float] = []
        for cell in selected:
            vals.extend(metric_values(cell, key))
        out[key] = mean(vals)
    scores: list[float] = []
    for cell in selected:
        scores.extend(cell.scores_attempted)
    out["score"] = mean(scores)
    return out


def delta_rows(cells: dict[tuple[str, str], Cell]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark in ["FCS", "ALE", "Research", "MLS"]:
        ours = pooled_group_metrics(cells, benchmark, "OURS")
        base = pooled_group_metrics(cells, benchmark, "BASE")
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "ours_tags": ", ".join(ours["tags"]),
            "base_tags": ", ".join(base["tags"]),
            "ours_n": f"{ours['parsed']}/{ours['attempted']}",
            "base_n": f"{base['parsed']}/{base['attempted']}",
        }
        for key in METRIC_KEYS + ["score"]:
            if ours.get(key) is None or base.get(key) is None:
                row[key] = None
            else:
                row[key] = float(ours[key]) - float(base[key])
        rows.append(row)
    return rows


def delta_table(rows: list[dict[str, Any]], include_tags: bool = True) -> str:
    if include_tags:
        header = [
            "benchmark",
            "ours_tags",
            "base_tags",
            "LOC",
            "AST_nodes",
            "cyclomatic",
            "imports",
            "fn_calls",
            "technique_kws",
            "score",
            "N_ours",
            "N_base",
        ]
        aligns = ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    else:
        header = [
            "benchmark",
            "LOC",
            "AST_nodes",
            "cyclomatic",
            "imports",
            "fn_calls",
            "technique_kws",
            "score",
            "N_ours",
            "N_base",
        ]
        aligns = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        values = [row["benchmark"]]
        if include_tags:
            values.extend([row["ours_tags"], row["base_tags"]])
        for key in METRIC_KEYS + ["score"]:
            digits = 3 if key == "score" else 2
            values.append(fmt_num(row.get(key), digits))
        values.extend([row["ours_n"], row["base_n"]])
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def direction(value: float | None, pos: str = "higher", neg: str = "lower") -> str:
    if value is None:
        return "n/a"
    if value > 0:
        return f"{pos} (+{fmt_num(value)})"
    if value < 0:
        return f"{neg} ({fmt_num(value)})"
    return "flat (+0.00)"


def headline(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "Complexity-only headline:",
        "Delta rows pool the parsed requested OURS tags and subtract the pooled parsed requested BASE tags for each benchmark.",
    ]
    recombine_votes = 0
    focused_votes = 0
    for row in rows:
        bench = row["benchmark"]
        loc = row.get("loc")
        cyc = row.get("cyclomatic")
        imports = row.get("imports")
        fn_calls = row.get("fn_calls")
        tech = row.get("technique_kws")
        libs_tech = None
        if imports is not None and tech is not None:
            libs_tech = imports + tech
        if fn_calls is not None and libs_tech is not None and (fn_calls > 0 or libs_tech > 0):
            recombine_votes += 1
        if loc is not None and cyc is not None and loc < 0 and cyc <= 0:
            focused_votes += 1
        lines.append(
            f"- {bench}: LOC {direction(loc, 'higher', 'lower')}, cyclomatic {direction(cyc, 'higher', 'lower')}, "
            f"imports+technique_kws {direction(libs_tech, 'more', 'fewer')}, fn_calls {direction(fn_calls, 'more', 'fewer')}."
        )
    if recombine_votes > focused_votes:
        lines.append(
            f"Overall, the numbers lean toward more building-block recombination in {recombine_votes}/4 benchmarks "
            f"(positive fn_calls or imports+technique_kws deltas), not a uniform low-LOC pattern."
        )
    elif focused_votes > recombine_votes:
        lines.append(
            f"Overall, the numbers lean toward more focused low-LOC moves in {focused_votes}/4 benchmarks "
            f"(negative LOC with non-higher cyclomatic), not a uniform recombination pattern."
        )
    else:
        lines.append(
            f"Overall, the numbers are mixed: {recombine_votes}/4 benchmarks show more building-block signals and "
            f"{focused_votes}/4 show lower-LOC/non-higher-cyclomatic movement."
        )
    lines.append("These are complexity directions only; they do not imply novelty or quality.")
    return lines


def skip_summary(cells: dict[tuple[str, str], Cell]) -> str:
    lines = [
        "| benchmark | model | attempted | parsed | empty/no_code | parse_errors | json_errors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ["FCS", "ALE", "Research", "MLS"]:
        bench_cells = [cell for (_model, bench), cell in cells.items() if bench == benchmark]
        bench_cells.sort(key=lambda c: (c.group, c.model))
        for cell in bench_cells:
            lines.append(
                f"| {benchmark} | {cell.group}:{cell.model} | {cell.attempted} | {cell.parsed} | "
                f"{cell.empty_code} | {cell.parse_errors} | {cell.json_errors} |"
            )
    return "\n".join(lines)


def source_summary(sources: list[ResolvedSource]) -> str:
    lines = [
        "| group | model | kind | resolved_source |",
        "|---|---|---|---|",
    ]
    for source in sources:
        lines.append(f"| {source.group} | {source.model} | {source.kind} | `{rel(source.path)}` |")
    missing = [source for source in sources if source.path is None]
    if missing:
        lines.append("")
        lines.append("Missing requested sources:")
        for source in missing:
            lines.append(f"- {source.group}:{source.model} ({source.kind}); tried:")
            for candidate in source.candidates:
                lines.append(f"  - `{rel(candidate)}`")
    else:
        lines.append("")
        lines.append("No requested source resolved as missing.")
    return "\n".join(lines)


def write_report(sources: list[ResolvedSource], cells: dict[tuple[str, str], Cell], rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Complexity Analysis: Innovation-Trained Qwen3.5-9B vs Base\n")
    lines.append("## 1. Methodology\n")
    lines.append(
        "This analysis was computed directly from raw sample JSONL files and MLS task logs under "
        "`/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs`. The script stripped any "
        "`<think>...</think>` block before fenced-code extraction."
    )
    lines.append("")
    lines.append(
        "For FCS rows, the extractor used the final fenced `cpp`/`c++` code block and split rows with "
        "`data_source=frontiercs`. For ALE and Research rows, it used the final fenced `python`/`py` "
        "code block and split rows with `data_source=alebench` or `frontiercs_research`. For MLS, "
        "each task log was one attempted solution; ANSI color codes were stripped and lines matching "
        "`^\\+\\s+\\d+\\s+\\|` were collected as agent-added Python code."
    )
    lines.append("")
    if RADON_AVAILABLE:
        lines.append("`radon` was importable, so Python cyclomatic complexity is the sum of `radon.cc_visit` block complexities.")
    else:
        lines.append(
            "`radon` was not importable, so Python cyclomatic complexity uses the requested manual proxy: "
            "`1 + if/for/while/ternary/except/comprehension decisions + and/or boolean joins`. "
            "C++ cyclomatic complexity uses `1 + if/for/while + && + ||`."
        )
    lines.append(
        "Python AST nodes are `len(ast.walk(ast.parse(code)))`; C++ AST_nodes is the requested brace-pair proxy. "
        "LOC counts non-blank, non-comment lines. Imports/includes and function calls are counted as distinct names. "
        "Advanced-technique keywords are counted as distinct listed keywords present case-insensitively in the code."
    )
    lines.append("")
    lines.append(
        "Solutions with empty extracted code or Python parse errors were skipped for complexity aggregation and counted below. "
        "Cells with fewer than 5 parsed solutions are flagged `LOW_N`. Standard deviations are sample standard deviations "
        "(0.00 for N=1)."
    )
    lines.append("")
    lines.append("Resolved raw sources:")
    lines.append("")
    lines.append(source_summary(sources))
    lines.append("")
    lines.append("Skip counts:")
    lines.append("")
    lines.append(skip_summary(cells))
    lines.append("")

    lines.append("## 2. Per-Benchmark Complexity Tables\n")
    for benchmark in ["FCS", "ALE", "Research", "MLS"]:
        lines.append(f"### {benchmark}\n")
        lines.append(table_for_benchmark(cells, benchmark))
        lines.append("")

    lines.append("## 3. OURS-BASE Delta Table\n")
    lines.append(
        "Deltas are pooled across parsed solutions from all requested tags in each group for that benchmark, then "
        "`OURS - BASE` is computed. Score is omitted for MLS because task logs do not carry per-task scores directly."
    )
    lines.append("")
    lines.append(delta_table(rows, include_tags=True))
    lines.append("")

    lines.append("## 4. Honest Headline\n")
    lines.extend(headline(rows))
    lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def print_stdout_summary(rows: list[dict[str, Any]]) -> None:
    print("OURS-BASE delta table (pooled parsed requested tags):")
    print(delta_table(rows, include_tags=False))
    print()
    for line in headline(rows):
        print(line)
    print()
    print(f"Wrote full report: {REPORT_PATH}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sources = resolve_sources()
    cells = compute_cells(sources)
    rows = delta_rows(cells)
    write_report(sources, cells, rows)
    print_stdout_summary(rows)


if __name__ == "__main__":
    main()
